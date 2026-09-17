// Run: node scripts/check-papers.mjs (optionally set CHROME_PATH or ARXIV_PAPER_DIR).
// Uses built-in Node APIs, local assets and a disposable Chrome profile.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
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
  && window.STUDY_PAPERS.length > 0
  && document.querySelectorAll('#paper-list .paper-card').length === window.STUDY_PAPERS.length`;
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
  // Open collapsed ancestors through their real summary control before clicking.
  // Calling HTMLElement.click() can otherwise make a hidden filter look usable.
  for (let attempt = 0; attempt < 10; attempt++) {
    const target = await evaluate(session, `(async () => {
      const element = document.querySelector(${JSON.stringify(selector)});
      if (!element) throw new Error('Missing control: ' + ${JSON.stringify(selector)});
      const parents = [];
      for (let node = element.parentElement; node; node = node.parentElement) {
        if (node.matches('details:not([open])') && !node.querySelector(':scope > summary')?.contains(element)) parents.unshift(node);
      }
      const control = parents[0]?.querySelector(':scope > summary') || element;
      const initial = control.getBoundingClientRect();
      if (initial.top < 0 || initial.bottom > innerHeight || initial.left < 0 || initial.right > innerWidth) {
        control.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'instant' });
      }
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const rect = control.getBoundingClientRect();
      const x = rect.left + rect.width / 2, y = rect.top + rect.height / 2;
      const hit = document.elementFromPoint(x, y);
      if (!rect.width || !rect.height || !(control === hit || control.contains(hit))) {
        throw new Error('Control is not pointer reachable: ' + ${JSON.stringify(selector)});
      }
      return { x, y, ancestor: parents.length > 0 };
    })()`);
    await cdp.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: target.x, y: target.y, button: 'left', clickCount: 1 }, session);
    await cdp.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: target.x, y: target.y, button: 'left', clickCount: 1 }, session);
    await evaluate(session, 'new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
    if (!target.ancestor) return;
  }
  throw new Error(`Unable to reveal control: ${selector}`);
}

async function search(session, text) {
  await click(session, '#paper-search');
  await evaluate(session, `(() => {
    const input = document.querySelector('#paper-search');
    input.value = ${JSON.stringify(text)};
    input.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
}


async function pressKey(session, key, shift = false) {
  const keyCode = { Tab: 9, Escape: 27, Enter: 13 }[key];
  const params = { key, code: key, windowsVirtualKeyCode: keyCode, modifiers: shift ? 8 : 0 };
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyDown', ...params, ...(key === 'Enter' ? {text:'\r', unmodifiedText:'\r'} : {}) }, session);
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
      const type = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.pdf': 'application/pdf' }[extname(path)];
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
  assert.equal(await evaluate(session, "document.querySelectorAll('h1').length"), 1);
  assert.equal(await evaluate(session, "document.querySelectorAll('[data-theme-toggle]').length"), 1);
  assert.equal(await evaluate(session, "document.querySelector('[data-theme-toggle]').closest('header.site-sidebar') !== null"), true,
    'The paper page must use the single shared-header theme control');
  assert.deepEqual(await evaluate(session, `[...document.scripts].map(script => new URL(script.src).pathname).filter(path => path.startsWith('/data/'))`), ['/data/papers.js', '/data/paper-analyses.js', '/data/paper-lessons.js']);
  await click(session, '#export-papers');
  assert.match(await evaluate(session, `document.querySelector('#paper-message').textContent`), /没有历史笔记/);
  assert.equal(await evaluate(session, `localStorage.getItem(${JSON.stringify(storageKey)})`), null);
  const papers = await evaluate(session, 'window.STUDY_PAPERS');
  const translated = papers.filter(paper => paper.collection === 'translated');
  const analysisPapers = papers.filter(paper => paper.collection !== 'translated');
  const firstAnalysis = analysisPapers[0];
  const manifest = JSON.parse(await readFile(join(root, 'data/paper-translations-manifest.json'), 'utf8'));
  assert.equal(manifest.version, 1);
  const translatedFiles = new Map(manifest.entries.map(entry => [entry.id, entry]));
  assert.equal(translatedFiles.size, translated.length, 'Every translated paper must have one PDF manifest entry');
  assert.deepEqual([...translatedFiles.keys()].sort(), translated.map(paper => paper.id).sort());
  const sourceRoot = process.env.ARXIV_PAPER_DIR || join(homedir(), 'Documents/hello_future/arxiv-paper');
  let verifySourceFiles = false;
  try { await readdir(sourceRoot); verifySourceFiles = true; }
  catch (error) {
    if (process.env.ARXIV_PAPER_DIR || !['ENOENT', 'EACCES', 'EPERM'].includes(error.code)) throw error;
  }
  assert.equal(papers.length, 51, 'The library must retain the original analyses, add four recommendation foundations and import all 39 completed translations');
  assert.equal(translated.length, 39);
  assert.equal(analysisPapers.length, 12);
  assert.equal(new Set(papers.map(paper => paper.id)).size, papers.length, 'Paper IDs must be unique');
  assert.deepEqual(await evaluate(session, visibleIds), translated.map(paper => paper.id),
    'The first visit should show the Chinese translations');
  assert.equal(await evaluate(session, `document.querySelector('[data-collection="translated"]').getAttribute('aria-pressed')`), 'true');
  assert.equal(await evaluate(session, `document.querySelectorAll('.paper-card[open]').length`), 0,
    'The library should initially show a scan-friendly list without expanding a paper');
  for (const collection of ['all', 'analysis', 'translated']) {
    await click(session, `[data-collection="${collection}"]`);
    const expected = collection === 'all' ? papers : collection === 'analysis' ? analysisPapers : translated;
    assert.deepEqual(await evaluate(session, visibleIds), expected.map(paper => paper.id));
  }
  for (const paper of translated) {
    const rendered = await evaluate(session, `(() => {
      const card = document.querySelector(${JSON.stringify(`[data-paper-id="${paper.id}"]`)});
      return {
        title: card.querySelector('.translation-heading h4')?.textContent,
        guide: card.querySelector('.paper-translation-guide')?.textContent,
        pdf: card.querySelector('.paper-source-links .chinese-pdf-link')?.href,
        links: Array.from(card.querySelectorAll('.paper-source-links a'), link => link.href),
        analysis: !!card.querySelector('.analysis-verdict'),
      };
    })()`);
    assert.ok(paper.zhTitle?.length > 0, `${paper.id}: translated title is required`);
    assert.equal(rendered.title, paper.zhTitle, `${paper.id}: translated title must be present in the reading guide`);
    for (const field of ['problem', 'method', 'reading']) {
      assert.ok(paper.guide?.[field]?.length > 0, `${paper.id}: missing reading guide ${field}`);
      assert.ok(rendered.guide?.includes(paper.guide[field]), `${paper.id}: guide ${field} must render`);
    }
    assert.equal(rendered.analysis, false, 'Translations should use the source-based reading guide');
    const pdfURL = new URL(paper.chinesePdf, `${base}/pages/papers.html`);
    assert.equal(rendered.pdf, pdfURL.href);
    assert.ok(rendered.links.includes(paper.url), `${paper.id}: English original link is required`);
    assert.equal(new URL(paper.url).protocol, 'https:');
    assert.ok(rendered.links.filter(link => new URL(link).origin === base).every(link => link === pdfURL.href),
      `${paper.id}: only the Chinese PDF should be hosted locally`);
    const entry = translatedFiles.get(paper.id);
    assert.equal(entry.path, pdfURL.pathname.slice(1), `${paper.id}: PDF URL must match the checked source manifest`);
    assert.equal(entry.bytes, paper.bytes);
    assert.equal(entry.pages, paper.pages);
    assert.match(entry.sha256, /^[a-f0-9]{64}$/);
    const pdf = await readFile(resolve(root, '.' + pdfURL.pathname));
    const digest = createHash('sha256').update(pdf).digest('hex');
    assert.equal(digest, entry.sha256, `${paper.id}: PDF must match the imported translation hash`);
    if (verifySourceFiles) {
      const source = await readFile(resolve(sourceRoot, entry.sourceFile));
      assert.equal(digest, createHash('sha256').update(source).digest('hex'),
        `${paper.id}: published Chinese PDF must be byte-identical to the completed local translation`);
    }
    assert.equal(pdf.subarray(0, 5).toString(), '%PDF-', `${paper.id}: local file must be a PDF`);
    assert.equal(pdf.length, paper.bytes, `${paper.id}: published size must match metadata`);
    assert.ok(Number.isInteger(paper.pages) && paper.pages > 0, `${paper.id}: page count must be present`);
    const response = await fetch(pdfURL);
    assert.equal(response.status, 200, `${paper.id}: Chinese PDF link must load`);
    assert.match(response.headers.get('content-type') || '', /^application\/pdf(?:;|$)/);
    assert.equal(createHash('sha256').update(Buffer.from(await response.arrayBuffer())).digest('hex'),
      digest, `${paper.id}: HTTP download must preserve the PDF`);
  }
  await search(session, translated[0].zhTitle);
  assert.deepEqual(await evaluate(session, visibleIds), [translated[0].id], 'Chinese title search should find the translation');
  await search(session, '');
  for (const category of ['ranking', 'sequential', 'generative']) {
    await click(session, `[data-category="${category}"]`);
    assert.equal(await evaluate(session, `document.querySelector('[data-category="${category}"]').getAttribute('aria-pressed')`), 'true',
      'Pointer selection inside the disclosure must commit the chosen category');
    assert.deepEqual(await evaluate(session, visibleIds), translated.filter(paper => paper.category === category).map(paper => paper.id));
  }
  await click(session, '[data-category="all"]');
  console.log(`PASS all 39 translation guides, Chinese PDF hashes/downloads${verifySourceFiles ? ' and source-file identity' : ''}, English original links, Chinese title search and collection filters`);
  await click(session, '[data-collection="analysis"]');
  assert.equal(await evaluate(session, `Array.from(document.querySelectorAll('.paper-card')).every(card =>
    card.tagName === 'DETAILS' && card.querySelector('summary') && !card.querySelector('.paper-notebook, textarea, [data-status-id], [data-template-id]'))`), true,
  'Each paper must use native details without the removed notebook');
  assert.equal(await evaluate(session, `document.querySelector('#paper-message').getAttribute('role')`), 'status');
  assert.deepEqual(await evaluate(session, 'Object.keys(window.PAPER_ANALYSES).sort()'), analysisPapers.map(paper => paper.id).sort(),
    'Every original analysis paper must have a locally loaded analysis');
  const analyses = await evaluate(session, 'window.PAPER_ANALYSES');
  const lessons = await evaluate(session, 'window.PAPER_LESSONS');
  for (const paper of analysisPapers) {
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
    assert.deepEqual(rendered.routes, ['overview', 'method', 'evidence', 'limits', ...(lessons[paper.id] ? ['study'] : [])].map(section => `#paper=${paper.id}&section=${section}`));
    for (const source of [...rendered.sources, ...rendered.rows.map(row => row.url)]) {
      assert.equal(new URL(source).protocol, 'https:', `${paper.id}: evidence sources must be usable HTTPS links`);
    }
  }
  console.log('PASS all 12 on-page analyses, method flows, formulas, evidence tables and original-source links');

  for (const [query, expected] of [['NF4', 'qlora'], ['MMLU', 'qlora'], ['WikiSQL', 'lora']]) {
    await search(session, query);
    assert.deepEqual(await evaluate(session, visibleIds), [expected], `Search must include analysis methods and experimental evidence: ${query}`);
  }
  await search(session, '');
  await click(session, '[data-category="architecture"]');
  for (const section of ['overview', 'method', 'evidence', 'limits', 'study']) {
    await click(session, `.analysis-nav a[href="#paper=${firstAnalysis.id}&section=${section}"]`);
    await verifySectionScroll(session, firstAnalysis.id, section);
    assert.equal(await evaluate(session, `document.querySelector('[data-category="architecture"]').getAttribute('aria-pressed')`), 'true',
      'Intra-paper navigation must preserve the selected category');
    assert.deepEqual(await evaluate(session, visibleIds), analysisPapers.filter(paper => paper.category === 'architecture').map(paper => paper.id));
  }
  const methodLink = `.analysis-nav a[href="#paper=${firstAnalysis.id}&section=method"]`;
  await click(session, methodLink);
  await verifySectionScroll(session, firstAnalysis.id, 'method');
  await evaluate(session, `window.scrollTo({ top: 0, behavior: 'instant' })`);
  await click(session, methodLink);
  await verifySectionScroll(session, firstAnalysis.id, 'method');
  await click(session, '[data-category="all"]');
  console.log('PASS method/evidence keyword search, section deep links and focus, same-hash scroll and preserved category filter');

  const expectedLessons = papers.map(paper => paper.id);
  assert.deepEqual(Object.keys(lessons).sort(), expectedLessons.sort());
  for (const id of expectedLessons) {
    const lesson = lessons[id];
    assert.equal(lesson.questions.length, 2);
    assert.ok(lesson.derivation.steps.length >= 3 && lesson.experiment.steps.length >= 3);
    for (const [related] of lesson.related) assert.ok(papers.some(paper => paper.id === related));
    const source = new URL(lesson.source.url, `${base}/pages/papers.html`);
    if (source.origin === base) assert.ok((await readFile(join(root, '.' + source.pathname))).length > 0);
    const rendered = await evaluate(session, `(() => {
      const lesson = document.getElementById('analysis-${id}-study');
      return { text: lesson.textContent, answers: [...lesson.querySelectorAll('.lesson-question details')].map(el => ({open:el.open, answer:el.querySelector('p').textContent})) };
    })()`);
    assert.ok(rendered.text.includes(lesson.derivation.formula));
    assert.deepEqual(rendered.answers.map(item => item.answer), lesson.questions.map(item => item.a));
    assert.ok(rendered.answers.every(item => !item.open), 'Answers stay folded until requested');
  }
  await click(session, '[data-collection="translated"]');
  await viewport(session, 1440);
  await click(session, '.learning-path-disclosure > summary');
  await screenshot(session, 'learning-routes-desktop.png', false);
  await click(session, '.paper-learning-routes a[href="#paper=din"]');
  await waitFor(session, `document.querySelector('#paper-din').open && !document.querySelector('#paper-din').hidden`);
  assert.equal(await evaluate(session, 'document.activeElement.parentElement.id'), 'paper-din');
  await click(session, '.paper-learning-routes a[href="#paper=tiger"]');
  await click(session, '#paper-tiger .analysis-nav a[href="#paper=tiger&section=study"]');
  await verifySectionScroll(session, 'tiger', 'study');
  await sectionScreenshot(session, 'tiger', 'study', 'tiger-study-desktop.png');
  const beforeStudy = await evaluate(session, `localStorage.getItem(${JSON.stringify(storageKey)})`);
  await evaluate(session, `document.querySelector('#paper-tiger .lesson-question summary').focus()`);
  await pressKey(session, 'Enter');
  assert.equal(await evaluate(session, `document.querySelector('#paper-tiger .lesson-question details').open`), true);
  await click(session, '#paper-tiger .lesson-code > summary');
  assert.equal(await evaluate(session, `document.querySelector('#paper-tiger .lesson-code').open`), true);
  await click(session, '#paper-tiger .lesson-related a[href="#paper=sasrec&section=study"]');
  await verifySectionScroll(session, 'sasrec', 'study');
  assert.equal(await evaluate(session, `localStorage.getItem(${JSON.stringify(storageKey)})`), beforeStudy, 'Learning controls must not alter existing notes');
  await click(session, '[data-collection="all"]');
  await search(session, '0.390625');
  assert.deepEqual(await evaluate(session, visibleIds), ['lora'], 'Search includes deep-study content');
  await search(session, '');
  await viewport(session, 320, 920);
  await evaluate(session, `location.hash = '#paper=tiger&section=study'`);
  await waitFor(session, `document.activeElement.id === 'analysis-tiger-study'`);
  await verifySectionScroll(session, 'tiger', 'study');
  await sectionScreenshot(session, 'tiger', 'study', 'tiger-study-320.png');
  assert.equal(await evaluate(session, `document.documentElement.scrollWidth <= innerWidth + 1`), true, 'Study text and code must reflow on narrow screens');
  await evaluate(session, `document.querySelector('#paper-tiger .lesson-code').scrollIntoView({block:'start',behavior:'instant'})`);
  await evaluate(session, `new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
  const codeShot = await cdp.send('Page.captureScreenshot', {format:'png'}, session);
  await writeFile(join(artifacts, 'tiger-code-320.png'), Buffer.from(codeShot.data,'base64'));
  await viewport(session, 1440);
  await evaluate(session, `StudyHubTheme.set('dark'); location.hash = '#paper=dpo&section=study'`);
  await waitFor(session, `document.activeElement.id === 'analysis-dpo-study'`);
  await sectionScreenshot(session, 'dpo', 'study', 'dpo-study-dark.png');
  await click(session, '#paper-dpo .lesson-question summary');
  await evaluate(session, `document.querySelector('#paper-dpo .lesson-questions').scrollIntoView({block:'start',behavior:'instant'}); new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
  const answerShot = await cdp.send('Page.captureScreenshot', {format:'png'}, session);
  await writeFile(join(artifacts, 'dpo-answer-dark.png'), Buffer.from(answerShot.data,'base64'));
  await evaluate(session, `StudyHubTheme.set('light')`);
  await click(session, '[data-collection="analysis"]');
  console.log('PASS 51 study lessons, 102 folded answers, route and related-paper navigation, keyboard reveal, searchable derivations and unchanged notes');


  for (const category of ['architecture', 'tuning', 'alignment', 'agents']) {
    await click(session, `[data-category="${category}"]`);
    assert.deepEqual(await evaluate(session, visibleIds), analysisPapers.filter(paper => paper.category === category).map(paper => paper.id));
  }
  await click(session, '[data-category="all"]');
  await search(session, `  ${firstAnalysis.title.toUpperCase()}  `);
  assert.deepEqual(await evaluate(session, visibleIds), [firstAnalysis.id], 'Search should ignore surrounding spaces and case');
  assert.match(await evaluate(session, `document.querySelector('#paper-count').textContent`), /1/);
  await search(session, 'no-paper-matches-this-regression-query');
  assert.deepEqual(await evaluate(session, visibleIds), []);
  assert.match(await evaluate(session, `document.querySelector('#paper-count').textContent`), /0/);
  await click(session, '#reset-filters');
  assert.equal(await evaluate(session, `document.querySelector('#paper-search').value`), '');
  assert.equal((await evaluate(session, visibleIds)).length, papers.length);
  console.log('PASS search, all subject filters, result count and empty-state reset');

  await click(session, '[data-collection="translated"]');
  const [first] = translated;
  await evaluate(session, `location.hash = ${JSON.stringify('#paper=' + first.id)}`);
  await waitFor(session, `location.hash === ${JSON.stringify('#paper=' + first.id)} && (() => {
    const card = document.querySelector(${JSON.stringify(`[data-paper-id="${first.id}"]`)});
    return !card.hidden && card.open && document.activeElement === card.querySelector('summary');
  })()`);
  const note = '我的阅读笔记：先理解问题，再核对实验。\n<script>alert("literal note")</script>';
  const legacy = { version: 1, records: { [first.id]: { status: 'done', note } } };
  await evaluate(session, `localStorage.setItem(${JSON.stringify(storageKey)}, ${JSON.stringify(JSON.stringify(legacy))})`);
  await cdp.send('Page.reload', {}, session);
  await waitFor(session, ready);
  assert.deepEqual(await evaluate(session, recordsExpression), legacy.records);
  assert.equal(await evaluate(session, `document.querySelectorAll('.paper-notebook, [data-note-id], [data-template-id], [data-status-id], [data-status-filter], #reading-progress').length`), 0);
  assert.equal(await evaluate(session, `document.querySelectorAll('.analysis-nav a[href$="section=notes"]').length`), 0);
  console.log('PASS notebook removed from all papers and historical records unchanged');

  await cdp.send('Browser.setDownloadBehavior', {
    behavior: 'allow', downloadPath: temporary, browserContextId,
  });
  legacy.records[first.id].note += '\n页面打开后更新的旧记录。';
  await evaluate(session, `localStorage.setItem(${JSON.stringify(storageKey)}, ${JSON.stringify(JSON.stringify(legacy))})`);
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
  assert.deepEqual({ status: backup.records[first.id].status, note: backup.records[first.id].note }, legacy.records[first.id]);
  assert.deepEqual(await evaluate(session, recordsExpression), legacy.records, 'Export must never rewrite stored notes');
  assert.equal(await evaluate(session, `document.querySelectorAll('#import-dialog, #import-papers, #choose-import, #storage-warning').length`), 0);
  console.log('PASS read-only archive download, unchanged stored data and removed import controls');

  const localLinks = await evaluate(session, `Array.from(document.querySelectorAll('a[href]'))
    .map(link => ({ text: link.textContent.trim(), url: link.href }))
    .filter(link => link.url.startsWith(location.origin + '/'))`);
  assert.ok(localLinks.some(link => new URL(link.url).pathname === '/index.html'), 'Paper topic must link home');
  assert.ok(localLinks.some(link => new URL(link.url).pathname === '/pages/interviews.html'),
    'Paper topic must link to the AI topic chooser');
  for (const pathname of new Set(localLinks.map(link => new URL(link.url).pathname))) {
    assert.equal((await fetch(base + pathname)).status, 200, `Local link must exist: ${pathname}`);
  }
  const home = await (await fetch(`${base}/index.html`)).text();
  assert.match(home, /href="pages\/papers\.html"/, 'Homepage must expose the reading topic');
  for (const page of ['transformer', 'sft', 'rl', 'agent']) {
    const interview = await openPage(`${base}/pages/interviews/${page}-interview.html`, undefined,
      `document.readyState === 'complete' && document.querySelectorAll('.sidebar-nav a').length === 4`);
    assert.equal(await evaluate(interview.sessionId,
      `Array.from(document.querySelectorAll('.sidebar-nav a')).find(link => new URL(link.href).pathname === '/pages/papers.html')?.href`), `${base}/pages/papers.html`,
    `${page} navigation must expose the reading topic`);
    assert.equal(await evaluate(interview.sessionId,
      "new URL(document.querySelector('.reader-back a').href).pathname"), '/pages/interviews.html',
    `${page} detail must return directly to its AI topic directory`);
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
      // Homepage layout remains part of the integration check; screenshots focus on the changed topic.
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
  await click(session, '[data-collection="all"]');
  await evaluate(session, `document.querySelectorAll('.paper-card').forEach(card => { card.open = true; })`);
  await viewport(session, 320);
  await evaluate(session, `document.querySelectorAll('.paper-card').forEach(card => { card.open = false; })`);
  await click(session, '[data-collection="translated"]');
  await click(session, `[data-paper-id="${first.id}"] summary`);
  for (const theme of ['light', 'dark']) {
    if (await evaluate(session, 'document.documentElement.dataset.theme') !== theme) await click(session, '[data-theme-toggle]');
    for (const width of [1440, 320]) {
      await viewport(session, width);
      await screenshot(session, `papers-${theme}-${width}.png`, false);
      await evaluate(session, `(() => {
        const guide = document.querySelector(${JSON.stringify(`[data-paper-id="${first.id}"] .paper-translation-guide`)});
        guide.scrollIntoView({ block: 'start', behavior: 'instant' });
        return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      })()`);
      const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' }, session);
      await writeFile(join(artifacts, `translation-${theme}-${width}-guide.png`), Buffer.from(data, 'base64'));
    }
  }
  await click(session, '[data-collection="analysis"]');
  await click(session, `[data-paper-id="${firstAnalysis.id}"] summary`);
  await sectionScreenshot(session, firstAnalysis.id, 'evidence', 'analysis-320-evidence.png');
  const tableScroll = await evaluate(session, `(() => {
    const region = document.querySelector(${JSON.stringify(`[data-paper-id="${firstAnalysis.id}"] .analysis-table-scroll`)});
    const initial = { width: region.clientWidth, scrollWidth: region.scrollWidth,
      label: region.getAttribute('aria-label'), tabindex: region.tabIndex };
    region.scrollLeft = region.scrollWidth;
    return { ...initial, scrolled: region.scrollLeft };
  })()`);
  assert.ok(tableScroll.scrollWidth > tableScroll.width && tableScroll.scrolled > 0,
    'Narrow evidence tables must still scroll inside their labelled region');
  assert.ok(tableScroll.label.length > 0 && tableScroll.tabindex === 0, 'Evidence scroll region must be keyboard reachable and labelled');
  await viewport(session, 320);
  console.log('PASS theme persistence, all 51 expanded cards without mobile overflow, translated guide views and existing evidence table scrolling');
  await cdp.send('Target.disposeBrowserContext', { browserContextId });

  const existingRecord = { status: 'reading', note: '升级前已保存的阅读笔记，应继续保留。' };
  const existing = await openPage(`${base}/pages/papers.html#paper=${firstAnalysis.id}&section=method`,
    `localStorage.setItem(${JSON.stringify(storageKey)}, ${JSON.stringify(JSON.stringify({ version: 1, records: { [firstAnalysis.id]: existingRecord } }))});`);
  assert.deepEqual((await evaluate(existing.sessionId, recordsExpression))[firstAnalysis.id], existingRecord);
  await verifySectionScroll(existing.sessionId, firstAnalysis.id, 'method');
  assert.ok((await evaluate(existing.sessionId, visibleIds)).includes(firstAnalysis.id), 'An existing analysis deep link must open across the default collection filter');
  await click(existing.sessionId, '[data-collection="translated"]');
  assert.deepEqual((await evaluate(existing.sessionId, recordsExpression))[firstAnalysis.id], existingRecord,
    'Navigating the updated library must preserve archived records');
  await cdp.send('Target.disposeBrowserContext', { browserContextId: existing.browserContextId });
  console.log('PASS existing analysis deep links and version 1 records remain intact after notebook removal');

  const file = await openPage(pathToFileURL(join(root, 'pages/papers.html')).href);
  assert.equal((await evaluate(file.sessionId, visibleIds)).length, translated.length);
  assert.equal(await evaluate(file.sessionId, `document.querySelectorAll('.paper-analysis .analysis-verdict').length`), analysisPapers.length,
    'All analysis content must also load from file://');
  assert.equal(await evaluate(file.sessionId, `document.querySelectorAll('.paper-translation-guide').length`), translated.length,
    'All translated guides must also load from file://');
  await search(file.sessionId, first.zhTitle);
  assert.deepEqual(await evaluate(file.sessionId, visibleIds), [first.id]);
  assert.equal(await evaluate(file.sessionId, `document.querySelectorAll('.paper-lesson').length`), papers.length);
  await cdp.send('Target.disposeBrowserContext', { browserContextId: file.browserContextId });
  console.log('PASS file:// assets, search and all lessons');

  const denied = await openPage(`${base}/pages/papers.html`, `Object.defineProperty(window, 'localStorage', {
    configurable: true, get() { throw new DOMException('regression: storage denied', 'SecurityError'); }
  });`);
  assert.equal(await evaluate(denied.sessionId, `document.querySelector('#paper-message').hidden`), true, 'Reading should not access archived storage');
  await click(denied.sessionId, '#export-papers');
  assert.match(await evaluate(denied.sessionId, `document.querySelector('#paper-message').textContent`), /无法读取/);
  await search(denied.sessionId, first.zhTitle);
  assert.deepEqual(await evaluate(denied.sessionId, visibleIds), [first.id]);
  const deniedTheme = await evaluate(denied.sessionId, 'document.documentElement.dataset.theme');
  await click(denied.sessionId, '[data-theme-toggle]');
  assert.notEqual(await evaluate(denied.sessionId, 'document.documentElement.dataset.theme'), deniedTheme);
  await cdp.send('Target.disposeBrowserContext', { browserContextId: denied.browserContextId });
  console.log('PASS denied storage notice and usable lessons, filters and theme');

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
