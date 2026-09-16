// Run: node scripts/check-layout.mjs [home topics papers transformer rl sft agent].
// Omit page names to check all pages; optionally set CHROME_PATH or BASE_URL for a deployed site.
// Exercises shared navigation and progressive disclosure in a disposable Chrome profile.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const chromePath = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const temporary = await mkdtemp(join(tmpdir(), 'layout-check-'));
const artifacts = await mkdtemp(join(tmpdir(), 'layout-check-artifacts-'));
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const errors = [], measurements = [];
let chrome, server, cdp;

class CdpPipe {
  constructor(child) {
    this.child = child; this.nextId = 0; this.pending = new Map(); this.buffer = Buffer.alloc(0);
    child.stdio[4].on('data', chunk => {
      this.buffer = Buffer.concat([this.buffer, chunk]);
      let boundary;
      while ((boundary = this.buffer.indexOf(0)) >= 0) {
        const packet = this.buffer.subarray(0, boundary).toString();
        this.buffer = this.buffer.subarray(boundary + 1);
        if (!packet) continue;
        const message = JSON.parse(packet);
        if (!message.id) { this.onEvent?.(message); continue; }
        const request = this.pending.get(message.id);
        if (!request) continue;
        this.pending.delete(message.id); clearTimeout(request.timer);
        if (message.error) request.reject(new Error(`${request.method}: ${message.error.message}`));
        else request.resolve(message.result);
      }
    });
    child.on('error', error => this.close(error));
    child.on('exit', (code, signal) => this.close(new Error(`Chrome exited: ${code} ${signal}`)));
  }
  send(method, params = {}, sessionId) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`${method} timed out`)); }, 30000);
      this.pending.set(id, { method, resolve, reject, timer });
      this.child.stdio[3].write(`${JSON.stringify({ id, method, params, sessionId })}\0`);
    });
  }
  close(error) {
    for (const request of this.pending.values()) { clearTimeout(request.timer); request.reject(error); }
    this.pending.clear();
  }
}
async function evaluate(session, expression) {
  const result = await cdp.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, session);
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  return result.result?.value;
}
async function waitFor(session, expression) {
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    try { if (await evaluate(session, expression)) return; }
    catch (error) { if (!/Execution context|Cannot find context|Inspected target navigated/.test(error.message)) throw error; }
    await delay(30);
  }
  throw new Error(`Page condition timed out: ${expression}`);
}
async function settle(session) {
  await evaluate(session, 'new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
}
async function click(session, selector, navigates = false) {
  const point = await evaluate(session, `(async () => {
    const element = document.querySelector(${JSON.stringify(selector)});
    if (!element) throw new Error('Missing control: ' + ${JSON.stringify(selector)});
    const initial = element.getBoundingClientRect();
    if (initial.top < 0 || initial.bottom > innerHeight || initial.left < 0 || initial.right > innerWidth) {
      element.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'instant' });
    }
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const rect = element.getBoundingClientRect();
    const x = rect.left + rect.width / 2, y = rect.top + rect.height / 2;
    const hit = document.elementFromPoint(x, y);
    if (!rect.width || !rect.height || !(element === hit || element.contains(hit))) throw new Error('Control is obscured: ' + ${JSON.stringify(selector)});
    return { x, y };
  })()`);
  await cdp.send('Input.dispatchMouseEvent', { type: 'mousePressed', ...point, button: 'left', clickCount: 1 }, session);
  await cdp.send('Input.dispatchMouseEvent', { type: 'mouseReleased', ...point, button: 'left', clickCount: 1 }, session);
  if (!navigates) await settle(session);
}
async function followLink(session, selector, pathname, ready) {
  await click(session, selector, true);
  await waitFor(session, `location.pathname === ${JSON.stringify(pathname)} && document.readyState === 'complete' && (${ready})`);
}
async function assertCurrentSection(session, pathname) {
  assert.equal(await evaluate(session, `new URL(document.querySelector('.sidebar-nav [aria-current="page"]').href).pathname`), pathname,
    'The current navigation item must identify the page section');
}
async function key(session, key, shift = false) {
  const keyCode = { Tab: 9, Enter: 13, Escape: 27, ' ': 32 }[key];
  for (const type of ['keyDown', 'keyUp']) await cdp.send('Input.dispatchKeyEvent', {
    type, key, code: key === ' ' ? 'Space' : key, windowsVirtualKeyCode: keyCode, modifiers: shift ? 8 : 0,
    ...(type === 'keyDown' && key === 'Enter' ? { text: '\r', unmodifiedText: '\r' } : {}),
  }, session);
  await settle(session);
}
async function capture(session, name, selector) {
  if (selector) await evaluate(session, `document.querySelector(${JSON.stringify(selector)}).scrollIntoView({ block: 'start', behavior: 'instant' })`);
  await settle(session);
  await evaluate(session, `Promise.race([
    Promise.allSettled(document.getAnimations().filter(animation =>
      animation.effect?.getComputedTiming().iterations !== Infinity).map(animation => animation.finished)),
    new Promise(resolve => setTimeout(resolve, 500))
  ])`);
  const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' }, session);
  await writeFile(join(artifacts, name + '.png'), Buffer.from(data, 'base64'));
}
async function assertNoOverflow(session, label) {
  const dimensions = await evaluate(session, `({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth })`);
  if (dimensions.scroll > dimensions.width + 1) {
    const elements = await evaluate(session, `Array.from(document.querySelectorAll('body *')).filter(element => {
      const rect = element.getBoundingClientRect(); return rect.width > 0 && rect.right > innerWidth + 1;
    }).slice(0, 15).map(element => ({ tag: element.tagName, id: element.id, class: element.className, right: element.getBoundingClientRect().right }))`);
    throw new Error(`${label}: horizontal overflow ${JSON.stringify(dimensions)}; ${JSON.stringify(elements)}`);
  }
}

try {
  server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      const path = resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
      if (!path.startsWith(root + sep)) throw new Error('Invalid path');
      const bytes = await readFile(path);
      response.writeHead(200, { 'Content-Type': { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.pdf': 'application/pdf' }[extname(path)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
      response.end(bytes);
    } catch { response.writeHead(404); response.end('Not found'); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  chrome = spawn(chromePath, [
    '--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
    '--disable-component-update', '--disable-default-apps', '--disable-sync', '--no-proxy-server',
    '--remote-debugging-pipe', `--user-data-dir=${join(temporary, 'profile')}`, 'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
  chrome.stderr.on('data', () => {});
  cdp = new CdpPipe(chrome);
  cdp.onEvent = message => {
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
    if (message.method === 'Network.responseReceived' && message.params.response.status >= 400 && !message.params.response.url.endsWith('/favicon.ico')) errors.push(`${message.params.response.status} ${message.params.response.url}`);
  };
  console.log(`Browser: ${(await cdp.send('Browser.getVersion')).product}`);
  const base = (process.env.BASE_URL || `http://127.0.0.1:${server.address().port}`).replace(/\/$/, '');
  const expectedRoutes = ['/index.html', '/pages/algorithms.html', '/pages/interviews.html', '/pages/papers.html'];
  assert.equal(new Set(expectedRoutes).size, 4, 'The four primary entrances must lead to distinct pages');
  for (const pathname of new Set(expectedRoutes.map(route => route.split('#')[0]))) assert.equal((await fetch(base + pathname)).status, 200, `Navigation destination must load: ${pathname}`);
  const pages = [
    { name: 'home', path: '/index.html', ready: "document.querySelector('#papers .paper-feature')", first: '.practice-card' },
    { name: 'topics', path: '/pages/interviews.html', ready: "document.querySelectorAll('#interviews .topic-card').length === 4", first: '#interviews .topic-card' },
    { name: 'papers', path: '/pages/papers.html', ready: "document.querySelectorAll('.paper-card').length === 47", first: '.paper-card:not([hidden]) > summary' },
    ...['transformer', 'rl', 'sft', 'agent'].map(name => ({ name, path: `/pages/interviews/${name}-interview.html`, ready: "document.querySelector('.q-card')", first: '.q-card > summary' })),
  ];
  const requested = process.argv.slice(2);
  for (const name of requested) assert.ok(pages.some(page => page.name === name), `Unknown page: ${name}`);
  for (const page of pages.filter(page => !requested.length || requested.includes(page.name))) {
    const { browserContextId } = await cdp.send('Target.createBrowserContext');
    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank', browserContextId });
    const { sessionId: session } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    await cdp.send('Page.enable', {}, session); await cdp.send('Runtime.enable', {}, session); await cdp.send('Network.enable', {}, session);
    await cdp.send('Page.navigate', { url: base + page.path }, session);
    await waitFor(session, `document.readyState === 'complete' && (${page.ready})`);
    const routes = await evaluate(session, `Array.from(document.querySelectorAll('.sidebar-nav .sidebar-link'), link => new URL(link.href).pathname + new URL(link.href).hash)`);
    assert.deepEqual(routes, expectedRoutes, `${page.name}: shared top navigation must expose the same four destinations`);
    assert.equal(await evaluate(session, "document.querySelectorAll('.sidebar-nav [aria-current]').length"), 1, `${page.name}: current section must be identifiable`);
    await assertCurrentSection(session, ['home', 'papers'].includes(page.name) ? page.path : '/pages/interviews.html');
    if (page.name === 'home') {
      assert.equal(await evaluate(session, "document.querySelectorAll('.learning-paths .path-card').length"), 3,
        'The overview must retain three main learning entrances');
      assert.equal(await evaluate(session, "document.querySelectorAll('#interviews .topic-card').length"), 0,
        'The topic directory must not remain duplicated on the overview');
    }
    if (page.name === 'topics') {
      const topics = await evaluate(session, `Array.from(document.querySelectorAll('#interviews .topic-card'), link => link.href)`);
      assert.equal(topics.length, 4);
      for (const url of topics) assert.equal((await fetch(url)).status, 200, 'AI topic entrance must load');
    }
    if (page.name === 'papers') {
      assert.equal(await evaluate(session, "document.querySelectorAll('.paper-card[open]').length"), 0);
      assert.equal(await evaluate(session, "document.querySelector('#paper-filters').open || document.querySelector('#paper-tools').open"), false);
    }
    for (const theme of ['light', 'dark']) {
      if (await evaluate(session, 'document.documentElement.dataset.theme') !== theme) await click(session, '[data-theme-toggle]');
      for (const width of [320, 390, 768, 1440]) {
        const height = width < 500 ? 844 : 960;
        await cdp.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false }, session);
        await evaluate(session, "window.scrollTo({ top: 0, behavior: 'instant' })"); await settle(session);
        await assertNoOverflow(session, `${page.name} ${theme} ${width}`);
        const metric = await evaluate(session, `(() => {
          const first = document.querySelector(${JSON.stringify(page.first)}).getBoundingClientRect();
          return { firstTop: Math.round(first.top), firstBottom: Math.round(first.bottom), viewportHeight: innerHeight,
            navigationVisible: Array.from(document.querySelectorAll('.sidebar-nav .sidebar-link')).every(link => {
              const rect = link.getBoundingClientRect(); return rect.width > 0 && rect.left >= 0 && rect.right <= innerWidth + 1 && rect.top >= 0 && rect.bottom <= innerHeight;
            }) };
        })()`);
        assert.equal(metric.navigationVisible, true, `${page.name} ${width}: every top-level navigation entrance must remain visible`);
        if (width < 500) assert.ok(metric.firstTop < metric.viewportHeight - 44, `${page.name} ${width}: the first content item must be discoverable on the initial screen (${metric.firstTop}px)`);
        measurements.push({ page: page.name, theme, width, ...metric });
        if ([320, 1440].includes(width)) await capture(session, `${page.name}-${theme}-${width}`);
        if (![320, 1440].includes(width)) continue;
        if (page.name === 'papers') {
          for (const selector of ['#paper-filters', '#paper-tools']) {
            await click(session, `${selector} > summary`);
            assert.equal(await evaluate(session, `document.querySelector(${JSON.stringify(selector)}).open`), true);
            await assertNoOverflow(session, `${page.name} ${selector} expanded ${width}`);
            await capture(session, `papers-${theme}-${width}-${selector.slice(1)}`);
            await evaluate(session, `document.querySelector(${JSON.stringify(selector + ' > summary')}).focus()`);
            await key(session, 'Enter');
            assert.equal(await evaluate(session, `document.querySelector(${JSON.stringify(selector)}).open`), false, 'A disclosure must close with the keyboard');
          }
        } else if (!['home', 'topics'].includes(page.name)) {
          if (!await evaluate(session, "document.querySelector('.toc-panel').open")) await click(session, '.toc-panel > summary');
          await assertNoOverflow(session, `${page.name} toc ${width}`);
          if (page.name === 'transformer') await capture(session, `${page.name}-${theme}-${width}-contents`, '.toc-panel');
          if (width < 901) await click(session, '.toc-panel > summary');
          await click(session, '.iv-toolbar > summary');
          assert.equal(await evaluate(session, "document.querySelector('.iv-toolbar').open"), true);
          await assertNoOverflow(session, `${page.name} tools ${width}`);
          if (page.name === 'transformer') await capture(session, `${page.name}-${theme}-${width}-tools`);
          await key(session, 'Escape');
          assert.equal(await evaluate(session, "document.querySelector('.iv-toolbar').open"), false);
          assert.equal(await evaluate(session, "document.activeElement.matches('.iv-toolbar > summary')"), true, 'Escape should restore the action-menu trigger');
        }
      }
    }
    // Verify keyboard activation of the first item without changing learning data.
    if (!['home', 'topics'].includes(page.name)) {
      const before = await evaluate(session, `(() => {
        const summary = document.querySelector(${JSON.stringify(page.first)});
        summary.focus(); return { open: summary.closest('details').open, focused: document.activeElement === summary };
      })()`);
      assert.equal(before.focused, true, `${page.name}: content disclosure must receive keyboard focus`);
      await key(session, 'Enter');
      assert.equal(await evaluate(session, `document.querySelector(${JSON.stringify(page.first)}).closest('details').open`), !before.open,
        `${page.name}: Enter must toggle the focused content disclosure`);
      await assertNoOverflow(session, `${page.name} expanded content`);
    }
    if (page.name === 'home') {
      const topicsReady = "document.querySelectorAll('#interviews .topic-card').length === 4";
      await followLink(session, '.interview-feature', '/pages/interviews.html', topicsReady);
      await assertCurrentSection(session, '/pages/interviews.html');
      const firstTopicPath = await evaluate(session, "new URL(document.querySelector('#interviews .topic-card').href).pathname");
      await followLink(session, '#interviews .topic-card', firstTopicPath, "document.querySelector('.q-card')");
      await assertCurrentSection(session, '/pages/interviews.html');
      await followLink(session, '.breadcrumbs a[href$="interviews.html"]', '/pages/interviews.html', topicsReady);
      await followLink(session, '.sidebar-nav .sidebar-link:first-child', '/index.html', page.ready);
      await assertCurrentSection(session, '/index.html');
      await followLink(session, '.sidebar-nav .sidebar-link:nth-child(3)', '/pages/interviews.html', topicsReady);
      await assertCurrentSection(session, '/pages/interviews.html');
      await cdp.send('Page.navigate', { url: base + '/index.html#interviews' }, session);
      await waitFor(session, `location.pathname === '/pages/interviews.html' && document.readyState === 'complete' && (${topicsReady})`);
      await assertCurrentSection(session, '/pages/interviews.html');
      await followLink(session, '.sidebar-nav .sidebar-link:first-child', '/index.html', page.ready);
      await evaluate(session, "location.hash = 'interviews'");
      await waitFor(session, `location.pathname === '/pages/interviews.html' && document.readyState === 'complete' && (${topicsReady})`);
      await assertCurrentSection(session, '/pages/interviews.html');
      console.log('PASS actual overview → topic directory → topic → breadcrumb navigation, distinct active sections and legacy hash redirects');
    }
    await cdp.send('Target.disposeBrowserContext', { browserContextId });
    console.log(`PASS ${page.name}: shared navigation, first content, 4 widths × 2 themes and relevant disclosures`);
  }
  assert.deepEqual(errors, [], 'No script or local-asset errors');
  await writeFile(join(artifacts, 'measurements.json'), JSON.stringify(measurements, null, 2));
  console.log('PASS all shared-layout checks');
} catch (error) {
  console.error(error.stack || error); process.exitCode = 1;
} finally {
  console.log(`Screenshots: ${artifacts}`);
  if (chrome?.pid && chrome.exitCode === null && chrome.signalCode === null) {
    const exited = new Promise(resolve => chrome.once('exit', resolve)); chrome.kill('SIGTERM'); await exited;
  }
  for (const stream of chrome?.stdio || []) stream?.destroy?.();
  if (server?.listening) { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
  await rm(temporary, { recursive: true, force: true });
}
