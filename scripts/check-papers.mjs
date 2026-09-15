// Run: node scripts/check-papers.mjs (optionally set CHROME_PATH).
// Uses built-in Node APIs, local assets and a disposable Chrome profile.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const chromePath = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const storageKey = 'study-hub:papers:v1';
const temporary = await mkdtemp(join(tmpdir(), 'papers-check-'));
const artifacts = await mkdtemp(join(tmpdir(), 'papers-check-artifacts-'));
const started = Date.now();
const browserErrors = [];
const externalRequests = [];
let chrome, cdp, server;

class CdpPipe {
  constructor(child) {
    this.nextId = 0;
    this.pending = new Map();
    this.buffer = Buffer.alloc(0);
    this.child = child;
    child.stdio[4].on('data', chunk => this.receive(chunk));
    child.stdio[4].on('error', error => this.close(error));
    child.on('error', error => this.close(error));
    child.on('exit', (code, signal) => this.close(new Error(`Chrome exited: ${code} ${signal}`)));
  }
  receive(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    let separator;
    while ((separator = this.buffer.indexOf(0)) >= 0) {
      const packet = this.buffer.subarray(0, separator).toString();
      this.buffer = this.buffer.subarray(separator + 1);
      if (!packet) continue;
      const message = JSON.parse(packet);
      if (!message.id) { this.onEvent?.(message); continue; }
      const pending = this.pending.get(message.id);
      if (!pending) continue;
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(`${pending.method}: ${message.error.message}`));
      else pending.resolve(message.result);
    }
  }
  send(method, params = {}, sessionId) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${method} timed out`));
      }, 30000);
      this.pending.set(id, { method, resolve, reject, timer });
      this.child.stdio[3].write(`${JSON.stringify({ id, method, params, sessionId })}\0`, error => {
        if (error) this.close(error);
      });
    });
  }
  close(error) {
    for (const pending of this.pending.values()) { clearTimeout(pending.timer); pending.reject(error); }
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
    if (await evaluate(session, expression)) return;
    await delay(50);
  }
  throw new Error(`Page condition timed out: ${expression}`);
}

const ready = `document.readyState === 'complete' && Array.isArray(window.STUDY_PAPERS)
  && document.querySelectorAll('#paper-list .paper-card').length === 8`;
const visibleIds = `Array.from(document.querySelectorAll('#paper-list .paper-card'))
  .filter(card => card.getClientRects().length && !card.hidden).map(card => card.dataset.paperId)`;
const recordsExpression = `(() => {
  const stored = JSON.parse(localStorage.getItem(${JSON.stringify(storageKey)}) || '{}');
  return stored.records || stored;
})()`;

async function openPage(url, beforeLoad, readyExpression = ready) {
  const { browserContextId } = await cdp.send('Target.createBrowserContext');
  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank', browserContextId });
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  await cdp.send('Page.enable', {}, sessionId);
  await cdp.send('Runtime.enable', {}, sessionId);
  await cdp.send('Network.enable', {}, sessionId);
  if (beforeLoad) await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: beforeLoad }, sessionId);
  await cdp.send('Page.navigate', { url }, sessionId);
  await waitFor(sessionId, readyExpression);
  return { sessionId, browserContextId };
}

async function click(session, selector) {
  await evaluate(session, `document.querySelector(${JSON.stringify(selector)}).click()`);
}

async function search(session, text) {
  await evaluate(session, `(() => {
    const input = document.querySelector('#paper-search');
    input.value = ${JSON.stringify(text)};
    input.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
}

async function setNote(session, id, text) {
  await evaluate(session, `(() => {
    const input = document.querySelector(${JSON.stringify(`[data-note-id="${id}"]`)});
    input.closest('details').open = true;
    input.value = ${JSON.stringify(text)};
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  })()`);
}

async function setStatus(session, id, status) {
  await evaluate(session, `(() => {
    const input = document.querySelector(${JSON.stringify(`input[name="status-${id}"][value="${status}"]`)});
    const card = input.closest('details');
    if (!card.open) card.querySelector('summary').click();
    input.click();
  })()`);
}

async function importFile(session, path) {
  const { root: document } = await cdp.send('DOM.getDocument', {}, session);
  const { nodeId } = await cdp.send('DOM.querySelector', { nodeId: document.nodeId, selector: '#import-papers' }, session);
  await cdp.send('DOM.setFileInputFiles', { nodeId, files: [path] }, session);
}

async function pressKey(session, key, shift = false) {
  const keyCode = { Tab: 9, Escape: 27 }[key];
  const params = { key, code: key, windowsVirtualKeyCode: keyCode, modifiers: shift ? 8 : 0 };
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyDown', ...params }, session);
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', ...params }, session);
}

async function screenshot(session, filename, fullPage = true) {
  await evaluate(session, `document.activeElement?.blur(); window.scrollTo({ top: 0, behavior: 'instant' });
    new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
  const visible = await cdp.send('Page.captureScreenshot', { format: 'png' }, session);
  await writeFile(join(artifacts, filename.replace('.png', '-viewport.png')), Buffer.from(visible.data, 'base64'));
  if (!fullPage) return;
  const { cssContentSize } = await cdp.send('Page.getLayoutMetrics', {}, session);
  const { data } = await cdp.send('Page.captureScreenshot', {
    format: 'png', captureBeyondViewport: true,
    clip: { x: 0, y: 0, width: cssContentSize.width, height: cssContentSize.height, scale: 1 },
  }, session);
  await writeFile(join(artifacts, filename), Buffer.from(data, 'base64'));
}

async function verifySectionScroll(session, id, section) {
  const hash = `#paper=${id}&section=${section}`;
  await waitFor(session, `location.hash === ${JSON.stringify(hash)} && (() => {
    const target = document.getElementById(${JSON.stringify(`analysis-${id}-${section}`)});
    const top = target.getBoundingClientRect().top + window.scrollY;
    const margin = parseFloat(getComputedStyle(target).scrollMarginTop) || 0;
    const padding = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
    const maximum = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const expected = Math.max(0, Math.min(top - margin - padding, maximum));
    return target.closest('details').open && Math.abs(window.scrollY - expected) <= 2
      && document.activeElement === target;
  })()`);
}

async function sectionScreenshot(session, id, section, filename) {
  await click(session, `.analysis-nav a[href="#paper=${id}&section=${section}"]`);
  await verifySectionScroll(session, id, section);
  await evaluate(session, `document.activeElement?.blur();
    new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
  const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' }, session);
  await writeFile(join(artifacts, filename), Buffer.from(data, 'base64'));
}

async function viewport(session, width) {
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    width, height: 960, deviceScaleFactor: 1, mobile: false,
  }, session);
  await evaluate(session, `new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
  const dimensions = await evaluate(session, `({ width: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth })`);
  if (dimensions.scroll > dimensions.width + 1) {
    const overflow = await evaluate(session, `Array.from(document.querySelectorAll('body *')).filter(element => {
      const rect = element.getBoundingClientRect();
      return rect.width > 0 && rect.right > document.documentElement.clientWidth + 1
        && !element.closest('.analysis-table-scroll') && !element.closest('.sidebar-nav');
    }).slice(0, 16).map(element => ({ tag: element.tagName, id: element.id, class: element.className,
      width: element.getBoundingClientRect().width, right: element.getBoundingClientRect().right,
      minWidth: getComputedStyle(element).minWidth }))`);
    console.error('Overflow elements:', JSON.stringify(overflow));
  }
  assert.ok(dimensions.scroll <= dimensions.width + 1,
    `${width}px viewport must not overflow horizontally: ${JSON.stringify(dimensions)}`);
}

try {
  server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      const path = resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
      if (!path.startsWith(root + sep)) throw new Error('Invalid path');
      const content = await readFile(path);
      const type = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css' }[extname(path)];
      response.writeHead(200, { 'Content-Type': type || 'application/octet-stream', 'Cache-Control': 'no-store' });
      response.end(content);
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
    if (message.method === 'Runtime.exceptionThrown') {
      const details = message.params.exceptionDetails;
      browserErrors.push(details.exception?.description || details.text);
    }
    if (message.method === 'Network.requestWillBeSent') {
      const url = message.params.request.url;
      if (/^https?:/.test(url) && new URL(url).hostname !== '127.0.0.1') externalRequests.push(url);
    }
    if (message.method === 'Network.responseReceived' && message.params.response.status >= 400
      && !message.params.response.url.endsWith('/favicon.ico')) {
      browserErrors.push(`${message.params.response.status} ${message.params.response.url}`);
    }
  };
  console.log(`Browser: ${(await cdp.send('Browser.getVersion')).product}`);
  const base = `http://127.0.0.1:${server.address().port}`;
  const { sessionId: session, browserContextId } = await openPage(`${base}/pages/papers.html`);
  const papers = await evaluate(session, 'window.STUDY_PAPERS');
  assert.equal(papers.length, 8);
  assert.equal(new Set(papers.map(paper => paper.id)).size, 8, 'Paper IDs must be unique');
  assert.deepEqual(await evaluate(session, visibleIds), papers.map(paper => paper.id));
  assert.equal(await evaluate(session, `Array.from(document.querySelectorAll('.paper-card')).every(card =>
    card.tagName === 'DETAILS' && card.querySelector('summary') && card.querySelector('textarea')?.labels.length > 0)`), true,
  'Each paper must use native details and a labelled note field');
  assert.equal(await evaluate(session, `document.querySelector('#paper-message').getAttribute('role')`), 'status');
  assert.deepEqual(await evaluate(session, 'Object.keys(window.PAPER_ANALYSES).sort()'), papers.map(paper => paper.id).sort(),
    'Every listed paper must have a locally loaded analysis');
  assert.deepEqual(await evaluate(session, `Array.from(document.querySelectorAll('.paper-card[open]')).map(card => card.dataset.paperId)`), [papers[0].id],
    'The first paper should open to its on-page analysis by default');
  const analyses = await evaluate(session, 'window.PAPER_ANALYSES');
  for (const paper of papers) {
    const rendered = await evaluate(session, `(() => {
      const card = document.querySelector(${JSON.stringify(`[data-paper-id="${paper.id}"]`)});
      const analysis = card.querySelector('.paper-analysis');
      return {
        article: analysis?.tagName, fallback: !!analysis?.querySelector('.paper-notice'),
        verdict: analysis?.querySelector('.analysis-verdict p')?.textContent,
        flow: Array.from(analysis?.querySelectorAll('.analysis-flow li strong') || [], item => item.textContent),
        formula: analysis?.querySelector('.analysis-formula code')?.textContent,
        rows: Array.from(analysis?.querySelectorAll('.analysis-evidence tbody tr') || [], row => ({
          label: row.querySelector('th')?.textContent, value: row.querySelector('td strong')?.textContent,
          url: row.querySelector('a.evidence-source')?.href,
        })),
        sources: Array.from(analysis?.querySelectorAll('.analysis-sources a') || [], link => link.href),
        original: Array.from(card.querySelectorAll('.paper-source-links a'), link => link.href),
        routes: Array.from(analysis?.querySelectorAll('.analysis-nav a') || [], link => link.hash),
      };
    })()`);
    const analysis = analyses[paper.id];
    assert.equal(rendered.article, 'ARTICLE', `${paper.id}: analysis must render rather than a fallback`);
    assert.equal(rendered.fallback, false);
    assert.equal(rendered.verdict, analysis.verdict);
    assert.ok(rendered.flow.length >= 3, `${paper.id}: method flow must have multiple steps`);
    assert.deepEqual(rendered.flow, analysis.flow.map(step => step.label));
    assert.equal(rendered.formula, analysis.formula.expression);
    assert.ok(rendered.rows.length >= 2, `${paper.id}: evidence must include at least two comparisons`);
    assert.deepEqual(rendered.rows, analysis.evidence.items.map(({ label, value, url }) => ({ label, value, url })));
    assert.deepEqual(rendered.sources, analysis.sources.map(source => source.url));
    assert.deepEqual(rendered.original, [paper.url, paper.pdf]);
    assert.deepEqual(rendered.routes, ['overview', 'method', 'evidence', 'limits', 'notes'].map(section => `#paper=${paper.id}&section=${section}`));
    for (const source of [...rendered.sources, ...rendered.rows.map(row => row.url)]) {
      assert.equal(new URL(source).protocol, 'https:', `${paper.id}: evidence sources must be usable HTTPS links`);
    }
  }
  console.log('PASS all 8 on-page analyses, method flows, formulas, evidence tables and original-source links');

  for (const [query, expected] of [['NF4', 'qlora'], ['MMLU', 'qlora'], ['WikiSQL', 'lora']]) {
    await search(session, query);
    assert.deepEqual(await evaluate(session, visibleIds), [expected], `Search must include analysis methods and experimental evidence: ${query}`);
  }
  await search(session, '');
  await click(session, '[data-category="architecture"]');
  for (const section of ['overview', 'method', 'evidence', 'limits', 'notes']) {
    await click(session, `.analysis-nav a[href="#paper=${papers[0].id}&section=${section}"]`);
    await verifySectionScroll(session, papers[0].id, section);
    assert.equal(await evaluate(session, `document.querySelector('[data-category="architecture"]').getAttribute('aria-pressed')`), 'true',
      'Intra-paper navigation must preserve the selected category');
    assert.deepEqual(await evaluate(session, visibleIds), papers.filter(paper => paper.category === 'architecture').map(paper => paper.id));
  }
  const methodLink = `.analysis-nav a[href="#paper=${papers[0].id}&section=method"]`;
  await click(session, methodLink);
  await verifySectionScroll(session, papers[0].id, 'method');
  await evaluate(session, `window.scrollTo({ top: 0, behavior: 'instant' })`);
  await click(session, methodLink);
  await verifySectionScroll(session, papers[0].id, 'method');
  await click(session, '[data-category="all"]');
  console.log('PASS method/evidence keyword search, section deep links and focus, same-hash scroll and preserved category filter');

  for (const category of ['architecture', 'tuning', 'alignment', 'agents']) {
    await click(session, `[data-category="${category}"]`);
    assert.deepEqual(await evaluate(session, visibleIds), papers.filter(paper => paper.category === category).map(paper => paper.id));
  }
  await click(session, '[data-category="all"]');
  await search(session, `  ${papers[0].title.toUpperCase()}  `);
  assert.deepEqual(await evaluate(session, visibleIds), [papers[0].id], 'Search should ignore surrounding spaces and case');
  assert.match(await evaluate(session, `document.querySelector('#paper-count').textContent`), /1/);
  await search(session, 'no-paper-matches-this-regression-query');
  assert.deepEqual(await evaluate(session, visibleIds), []);
  assert.match(await evaluate(session, `document.querySelector('#paper-count').textContent`), /0/);
  await click(session, '#reset-filters');
  assert.equal(await evaluate(session, `document.querySelector('#paper-search').value`), '');
  assert.equal((await evaluate(session, visibleIds)).length, 8);
  console.log('PASS search, all subject filters, result count and empty-state reset');

  const [first, second] = papers;
  const note = '我的阅读笔记：先理解问题，再核对实验。\n<script>alert("literal note")</script>';
  await setStatus(session, first.id, 'reading');
  await setNote(session, first.id, note);
  await waitFor(session, `${recordsExpression}[${JSON.stringify(first.id)}]?.note === ${JSON.stringify(note)}`);
  await cdp.send('Page.reload', {}, session);
  await waitFor(session, ready);
  assert.equal(await evaluate(session, `document.querySelector(${JSON.stringify(`input[name="status-${first.id}"]:checked`)}).value`), 'reading');
  assert.equal(await evaluate(session, `document.querySelector(${JSON.stringify(`[data-note-id="${first.id}"]`)}).value`), note);
  await click(session, '[data-status-filter="reading"]');
  assert.deepEqual(await evaluate(session, visibleIds), [first.id]);
  await click(session, '[data-status-filter="done"]');
  assert.deepEqual(await evaluate(session, visibleIds), []);
  await click(session, '#reset-filters');
  await setStatus(session, first.id, 'done');
  assert.match(await evaluate(session, `document.querySelector('#completed-count').textContent`), /1/);
  await click(session, '[data-status-filter="done"]');
  assert.deepEqual(await evaluate(session, visibleIds), [first.id]);
  await click(session, '[data-status-filter="all"]');
  console.log('PASS status filters, completion count and exact note persistence across reload');

  await cdp.send('Browser.setDownloadBehavior', {
    behavior: 'allow', downloadPath: temporary, browserContextId,
  });
  await click(session, '#export-papers');
  let backup;
  for (let attempt = 0; attempt < 100 && !backup; attempt++) {
    for (const filename of await readdir(temporary)) {
      if (filename.endsWith('.json')) {
        try { backup = JSON.parse(await readFile(join(temporary, filename), 'utf8')); } catch { /* Download is still writing. */ }
      }
    }
    if (!backup) await delay(50);
  }
  assert.equal(backup?.version, 1, 'Export must download a versioned JSON backup');
  assert.deepEqual({ status: backup.records[first.id].status, note: backup.records[first.id].note }, { status: 'done', note });
  const beforeImport = await evaluate(session, recordsExpression);
  const invalidPaths = ['invalid-json.json', 'invalid-record.json', 'invalid-status-type.json'].map(name => join(temporary, name));
  await writeFile(invalidPaths[0], '{invalid JSON');
  await writeFile(invalidPaths[1], JSON.stringify({ version: 1, records: {
    [first.id]: { status: 'unread', note: 'This partial import must never be saved.' },
    [second.id]: { status: 'invalid-status', note: '' },
  } }));
  await writeFile(invalidPaths[2], JSON.stringify({ version: 1, records: {
    [first.id]: { status: ['done'], note: 'An array must not be coerced into a valid status.' },
  } }));
  for (const path of invalidPaths) {
    await evaluate(session, `document.querySelector('#paper-message').textContent = ''`);
    await importFile(session, path);
    await waitFor(session, `document.querySelector('#paper-message').textContent.trim().length > 0`);
    assert.match(await evaluate(session, `document.querySelector('#paper-message').textContent`), /失败|无效|不支持|无法|格式|错误|不合法/);
    assert.equal(await evaluate(session, `document.querySelector('#import-dialog').open`), false,
      'Invalid backup must not open a confirmation dialog');
    assert.deepEqual(await evaluate(session, recordsExpression), beforeImport, 'Invalid import must preserve every previous record');
  }
  const mergedNote = '补充记录：比较方法的假设与局限。';
  const validPath = join(temporary, 'merge.json');
  await writeFile(validPath, JSON.stringify({ version: 1, records: { [second.id]: { status: 'reading', note: mergedNote } } }));
  await importFile(session, validPath);
  await waitFor(session, `document.querySelector('#import-dialog').open`);
  assert.deepEqual(await evaluate(session, recordsExpression), beforeImport, 'Import preview must not change records');
  await click(session, '#cancel-import');
  assert.deepEqual(await evaluate(session, recordsExpression), beforeImport, 'Cancelled import must retain records');
  await importFile(session, validPath);
  await waitFor(session, `document.querySelector('#import-dialog').open`);
  await click(session, '#confirm-import');
  await waitFor(session, `${recordsExpression}[${JSON.stringify(second.id)}]?.note === ${JSON.stringify(mergedNote)}`);
  const merged = await evaluate(session, recordsExpression);
  assert.deepEqual(merged[first.id], beforeImport[first.id], 'Import must retain records absent from the backup');
  assert.equal(merged[second.id].status, 'reading');
  assert.equal(await evaluate(session, `document.querySelector(${JSON.stringify(`[data-note-id="${second.id}"]`)}).value`), mergedNote);
  const conflictingPath = join(temporary, 'conflicting-import.json');
  await writeFile(conflictingPath, JSON.stringify({ version: 1, records: {
    [first.id]: { status: 'unread', note: 'Pressing Escape must never apply this replacement.' },
  } }));
  await importFile(session, conflictingPath);
  await waitFor(session, `document.querySelector('#import-dialog').open`);
  assert.equal(await evaluate(session, 'document.activeElement.id'), 'cancel-import', 'Import should initially focus Cancel');
  await pressKey(session, 'Tab');
  assert.equal(await evaluate(session, 'document.activeElement.id'), 'confirm-import');
  await pressKey(session, 'Tab', true);
  assert.equal(await evaluate(session, 'document.activeElement.id'), 'cancel-import');
  await pressKey(session, 'Escape');
  await waitFor(session, `!document.querySelector('#import-dialog').open && document.activeElement.id === 'choose-import'`);
  assert.deepEqual(await evaluate(session, recordsExpression), merged,
    'Escape after a previously confirmed import must cancel, even when the previous returnValue was confirm');
  console.log('PASS real JSON download, malformed/type-invalid imports, preview/cancel/merge and keyboard Escape after confirm');

  await click(session, `[data-template-id="${first.id}"]`);
  const withTemplate = await evaluate(session, `document.querySelector(${JSON.stringify(`[data-note-id="${first.id}"]`)}).value`);
  assert.ok(withTemplate.startsWith(note + '\n\n'), 'Template must append without replacing the existing note');
  assert.match(withTemplate, /研究问题[\s\S]*核心方法[\s\S]*实验证据[\s\S]*局限与疑问/);
  assert.equal((await evaluate(session, recordsExpression))[first.id].note, withTemplate);
  console.log('PASS note template preserves existing text and saves the appended content');

  const localLinks = await evaluate(session, `Array.from(document.querySelectorAll('a[href]'))
    .map(link => ({ text: link.textContent.trim(), url: link.href }))
    .filter(link => link.url.startsWith(location.origin + '/'))`);
  assert.ok(localLinks.some(link => new URL(link.url).pathname === '/index.html'), 'Paper topic must link home');
  assert.ok(localLinks.some(link => new URL(link.url).pathname.startsWith('/pages/interviews/')), 'Paper topic must link to interview learning');
  for (const pathname of new Set(localLinks.map(link => new URL(link.url).pathname))) {
    assert.equal((await fetch(base + pathname)).status, 200, `Local link must exist: ${pathname}`);
  }
  const home = await (await fetch(`${base}/index.html`)).text();
  assert.match(home, /href="pages\/papers\.html"/, 'Homepage must expose the reading topic');
  for (const page of ['transformer', 'sft', 'rl', 'agent']) {
    const interview = await openPage(`${base}/pages/interviews/${page}-interview.html`, undefined,
      `document.readyState === 'complete' && document.querySelector('.sidebar-nav a[href="../papers.html"]')`);
    assert.equal(await evaluate(interview.sessionId,
      `document.querySelector('.sidebar-nav a[href="../papers.html"]').href`), `${base}/pages/papers.html`,
    `${page} sidebar must expose the reading topic`);
    await cdp.send('Target.disposeBrowserContext', { browserContextId: interview.browserContextId });
  }
  console.log('PASS homepage, paper topic and interview navigation');

  const overview = await openPage(`${base}/index.html`, undefined,
    `document.readyState === 'complete' && document.querySelector('#papers .paper-feature')`);
  for (const theme of ['light', 'dark']) {
    if (await evaluate(overview.sessionId, 'document.documentElement.dataset.theme') !== theme) {
      await click(overview.sessionId, '[data-theme-toggle]');
    }
    for (const width of [1440, 390]) {
      await viewport(overview.sessionId, width);
      await screenshot(overview.sessionId, `home-${theme}-${width}.png`);
    }
  }
  await cdp.send('Target.disposeBrowserContext', { browserContextId: overview.browserContextId });
  console.log('PASS homepage 390px/1440px layout in both themes');

  const originalTheme = await evaluate(session, 'document.documentElement.dataset.theme');
  await click(session, '[data-theme-toggle]');
  assert.notEqual(await evaluate(session, 'document.documentElement.dataset.theme'), originalTheme);
  await cdp.send('Page.reload', {}, session);
  await waitFor(session, ready);
  assert.notEqual(await evaluate(session, 'document.documentElement.dataset.theme'), originalTheme, 'Theme choice should survive reload');
  await evaluate(session, `document.querySelectorAll('.paper-card').forEach(card => { card.open = true; })`);
  await viewport(session, 320);
  await evaluate(session, `document.querySelectorAll('.paper-card').forEach((card, index) => { card.open = index === 0; })`);
  for (const theme of ['light', 'dark']) {
    if (await evaluate(session, 'document.documentElement.dataset.theme') !== theme) await click(session, '[data-theme-toggle]');
    await viewport(session, 1440);
    await screenshot(session, `papers-${theme}-1440.png`, false);
    for (const section of ['method', 'evidence', 'notes']) {
      await sectionScreenshot(session, first.id, section, `analysis-${theme}-1440-${section}.png`);
    }
    await viewport(session, 320);
    await screenshot(session, `papers-${theme}-320-expanded.png`, false);
    for (const section of ['method', 'evidence', 'notes']) {
      await sectionScreenshot(session, first.id, section, `analysis-${theme}-320-${section}.png`);
    }
    const tableScroll = await evaluate(session, `(() => {
      const region = document.querySelector(${JSON.stringify(`[data-paper-id="${first.id}"] .analysis-table-scroll`)});
      const initial = { width: region.clientWidth, scrollWidth: region.scrollWidth,
        label: region.getAttribute('aria-label'), tabindex: region.tabIndex };
      region.scrollLeft = region.scrollWidth;
      return { ...initial, scrolled: region.scrollLeft };
    })()`);
    assert.ok(tableScroll.scrollWidth > tableScroll.width && tableScroll.scrolled > 0,
      'Narrow evidence tables must scroll inside their labelled region');
    assert.ok(tableScroll.label.length > 0 && tableScroll.tabindex === 0, 'Evidence scroll region must be keyboard reachable and labelled');
    await sectionScreenshot(session, first.id, 'evidence', `analysis-${theme}-320-evidence-scrolled.png`);
    await evaluate(session, `document.querySelector(${JSON.stringify(`[data-paper-id="${first.id}"] .analysis-table-scroll`)}).scrollLeft = 0`);
    await viewport(session, 320);
  }
  console.log('PASS theme toggle/persistence, method/evidence/note views and contained mobile table scrolling');
  await cdp.send('Target.disposeBrowserContext', { browserContextId });

  const existingRecord = { status: 'reading', note: '升级前已保存的阅读笔记，应继续保留。' };
  const existing = await openPage(`${base}/pages/papers.html`,
    `localStorage.setItem(${JSON.stringify(storageKey)}, ${JSON.stringify(JSON.stringify({ version: 1, records: { [first.id]: existingRecord } }))});`);
  assert.equal(await evaluate(existing.sessionId,
    `document.querySelector(${JSON.stringify(`input[name="status-${first.id}"]:checked`)}).value`), existingRecord.status);
  assert.equal(await evaluate(existing.sessionId,
    `document.querySelector(${JSON.stringify(`[data-note-id="${first.id}"]`)}).value`), existingRecord.note);
  assert.deepEqual((await evaluate(existing.sessionId, recordsExpression))[first.id], existingRecord);
  await cdp.send('Target.disposeBrowserContext', { browserContextId: existing.browserContextId });
  console.log('PASS pre-existing version 1 reading records remain intact with the expanded analyses');

  const file = await openPage(pathToFileURL(join(root, 'pages/papers.html')).href);
  assert.equal((await evaluate(file.sessionId, visibleIds)).length, 8);
  assert.equal(await evaluate(file.sessionId, `document.querySelectorAll('.paper-analysis .analysis-verdict').length`), 8,
    'All analysis content must also load from file://');
  await search(file.sessionId, papers[0].title);
  assert.deepEqual(await evaluate(file.sessionId, visibleIds), [first.id]);
  await setNote(file.sessionId, first.id, '本地文件阅读笔记');
  await cdp.send('Page.reload', {}, file.sessionId);
  await waitFor(file.sessionId, ready);
  assert.equal(await evaluate(file.sessionId, `document.querySelector(${JSON.stringify(`[data-note-id="${first.id}"]`)}).value`), '本地文件阅读笔记');
  await cdp.send('Target.disposeBrowserContext', { browserContextId: file.browserContextId });
  console.log('PASS file:// assets, search and note persistence');

  const denied = await openPage(`${base}/pages/papers.html`, `Object.defineProperty(window, 'localStorage', {
    configurable: true, get() { throw new DOMException('regression: storage denied', 'SecurityError'); }
  });`);
  await waitFor(denied.sessionId, `document.querySelector('#storage-warning')?.getClientRects().length > 0`);
  assert.match(await evaluate(denied.sessionId, `document.querySelector('#storage-warning').textContent`), /保存|存储|浏览器|备份/);
  await setStatus(denied.sessionId, first.id, 'done');
  await setNote(denied.sessionId, first.id, '存储不可用时，当前页面仍可编辑。');
  await click(denied.sessionId, '[data-status-filter="done"]');
  assert.deepEqual(await evaluate(denied.sessionId, visibleIds), [first.id]);
  const deniedTheme = await evaluate(denied.sessionId, 'document.documentElement.dataset.theme');
  await click(denied.sessionId, '[data-theme-toggle]');
  assert.notEqual(await evaluate(denied.sessionId, 'document.documentElement.dataset.theme'), deniedTheme);
  await cdp.send('Target.disposeBrowserContext', { browserContextId: denied.browserContextId });
  console.log('PASS denied storage notice and usable in-session notes, filters and theme');

  assert.deepEqual(browserErrors, [], 'No browser script or local asset errors');
  assert.deepEqual(externalRequests, [], 'Reading page must not require external requests');
  console.log(`PASS all paper checks (${((Date.now() - started) / 1000).toFixed(1)}s)`);
} catch (error) {
  console.error(error.stack || error);
  process.exitCode = 1;
} finally {
  console.log(`Screenshots: ${artifacts}`);
  if (chrome?.pid && chrome.exitCode === null && chrome.signalCode === null) {
    const exited = new Promise(resolve => chrome.once('exit', resolve));
    chrome.kill('SIGTERM');
    await exited;
  }
  for (const stream of chrome?.stdio || []) stream?.destroy?.();
  if (server?.listening) {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
  await rm(temporary, { recursive: true, force: true });
}
