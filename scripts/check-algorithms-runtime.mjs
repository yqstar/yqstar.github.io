// Run: node scripts/check-algorithms-runtime.mjs (optionally set CHROME_PATH).
// Uses a disposable Chrome profile; no npm dependencies or network access required.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { copyFile, mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const chromePath = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
// Keep the reported user solution independent of future reference-solution changes.
const moveZeroesSource = `class Solution:
    def moveZeroes(self, nums):
        write = 0
        for x in nums:
            if x != 0:
                nums[write] = x
                write += 1
        nums[write:] = [0] * (len(nums) - write)`;

class CdpPipe {
  constructor(child) {
    this.child = child;
    this.nextId = 0;
    this.pending = new Map();
    this.buffer = Buffer.alloc(0);
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

let chrome, cdp, server;
const temporary = await mkdtemp(join(tmpdir(), 'algorithms-runtime-'));
const workerSessions = new Set();
const externalRequests = [];
const workerSetupErrors = [];
// Deterministic compatibility simulation: JavaScript Blob URLs cannot be imported.
// This is not a Safari/WebKit test; it reproduces that module-loading constraint.
const unavailableJavaScriptBlobs = `(() => {
  const create = URL.createObjectURL.bind(URL);
  URL.createObjectURL = blob => {
    const url = create(blob);
    if (/javascript/.test(blob.type)) URL.revokeObjectURL(url);
    return url;
  };
})()`;
async function evaluate(session, expression) {
  const result = await cdp.send('Runtime.evaluate', {
    expression, awaitPromise: true, returnByValue: true,
  }, session);
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  }
  return result.result?.value;
}
async function waitFor(session, expression) {
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    if (await evaluate(session, expression)) return;
    await delay(100);
  }
  throw new Error(`Page condition timed out: ${expression}`);
}

try {
  server = createServer(async (request, response) => {
    try {
      const path = resolve(root, '.' + decodeURIComponent(new URL(request.url, 'http://localhost').pathname));
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
    if (message.method === 'Target.attachedToTarget') {
      const { sessionId, targetInfo } = message.params;
      // Pause each real Worker before startup, then block all HTTP(S) fetches/imports.
      // Embedded blob:/data: assets remain available, including the Wasm URL mapping.
      (async () => {
        if (targetInfo.type === 'worker') {
          workerSessions.add(sessionId);
          await cdp.send('Network.enable', {}, sessionId);
          await cdp.send('Network.setBlockedURLs', { urls: ['http://*', 'https://*'] }, sessionId);
          const injected = await cdp.send('Runtime.evaluate', { expression: unavailableJavaScriptBlobs }, sessionId);
          if (injected.exceptionDetails) throw new Error('Could not inject JavaScript Blob compatibility simulation');
        }
        await cdp.send('Runtime.runIfWaitingForDebugger', {}, sessionId);
      })().catch(error => workerSetupErrors.push(error.message));
    }
    if (message.method === 'Network.requestWillBeSent' && workerSessions.has(message.sessionId)
      && /^https?:/.test(message.params.request.url)) externalRequests.push(message.params.request.url);
  };
  const version = await cdp.send('Browser.getVersion');
  console.log(`Browser: ${version.product}`);
  const standalone = join(temporary, 'algorithms.html');
  await copyFile(join(root, 'pages/algorithms.html'), standalone);
  const targets = [
    ['HTTP', `http://127.0.0.1:${server.address().port}/pages/algorithms.html`],
    ['standalone file', pathToFileURL(standalone).href],
  ];
  for (const [label, url] of targets) {
    const { browserContextId } = await cdp.send('Target.createBrowserContext');
    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank', browserContextId });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    const workerCount = workerSessions.size;
    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: true, flatten: true }, sessionId);
    await cdp.send('Page.navigate', { url }, sessionId);
    await waitFor(sessionId, `document.readyState === 'complete' && typeof ensureJudge === 'function'`);
    assert.equal(await evaluate(sessionId, `(async () => {
      await ensureJudge(); return elements.runtime_status.classList.contains('ready');
    })()`), true, `${label}: Python becomes ready`);
    console.log(`PASS ${label}: Python startup; worker HTTP(S) blocked; inaccessible JS Blob simulation`);

    const results = await evaluate(sessionId, `(async () => {
      const { code, tests, note, complexity, ...meta } = SOLUTIONS['move-zeroes'];
      window.runtimeTestPayload = { mode: 'core', userCode: ${JSON.stringify(moveZeroesSource)}, referenceCode: code, meta,
        cases: tests.map(value => ({ value })) };
      return evaluate(runtimeTestPayload, 6000);
    })()`);
    assert.equal(results.passed, true, `${label}: mutation cases pass`);
    assert.deepEqual(results.results.map(item => JSON.parse(item.actual)), [[1, 3, 12, 0, 0], [0], [1, 2, 3], [1, 0, 0]]);
    await evaluate(sessionId, `openProblem('move-zeroes'); elements.code_editor.value = runtimeTestPayload.userCode;
      elements.code_editor.dispatchEvent(new Event('input', { bubbles: true })); elements.run_button.click();`);
    await waitFor(sessionId, `!evaluationInProgress && testConsoleState.evaluation !== null`);
    assert.equal(await evaluate(sessionId, `testConsoleState.evaluation?.passed && elements.result_panel.textContent.includes('运行通过')`), true);
    console.log(`PASS ${label}: move-zeroes mutations and Run button results`);

    const formatted = await evaluate(sessionId, `formatPythonSource('x=1+2\\nprint( x )\\n')`);
    assert.equal(formatted.ok, true);
    assert.equal(formatted.changed, true);
    assert.equal(formatted.code, 'x = 1 + 2\nprint(x)\n');
    console.log(`PASS ${label}: embedded Python formatter`);

    const retry = await evaluate(sessionId, `(async () => {
      (await ensureJudge()).terminate(); judgeReady = null;
      const NativeWorker = window.Worker;
      window.Worker = class extends NativeWorker { constructor() { throw new Error('regression: startup failure'); } };
      let failure;
      try { await ensureJudge(); } catch (error) { failure = error.message; } finally { window.Worker = NativeWorker; }
      const cleared = judgeReady === null;
      await ensureJudge();
      return { failure, cleared, passed: (await evaluate(runtimeTestPayload, 6000)).passed };
    })()`);
    assert.match(retry.failure || '', /regression: startup failure/);
    assert.equal(retry.cleared, true);
    assert.equal(retry.passed, true);
    console.log(`PASS ${label}: startup failure clears cached promise and permits retry`);

    const recovery = await evaluate(sessionId, `(async () => {
      let failure;
      try { await evaluate({ ...runtimeTestPayload, userCode: 'while True:\\n    pass' }, 200); }
      catch (error) { failure = error.message; }
      const cleared = judgeReady === null && pendingPythonRequest === null;
      return { failure, cleared, passed: (await evaluate(runtimeTestPayload, 6000)).passed };
    })()`);
    assert.match(recovery.failure || '', /已终止 Python 进程/);
    assert.equal(recovery.cleared, true);
    assert.equal(recovery.passed, true);
    assert.equal(workerSessions.size - workerCount, 3, `${label}: three real Worker lifetimes were network-blocked`);
    assert.deepEqual(workerSetupErrors, []);
    assert.deepEqual(externalRequests, [], 'Runtime must not request external HTTP(S) resources');
    console.log(`PASS ${label}: infinite loop times out and a fresh Worker runs successfully`);
    await cdp.send('Target.disposeBrowserContext', { browserContextId });
  }
  console.log('PASS all runtime checks (HTTP and standalone file; no external runtime requests)');
} catch (error) {
  console.error(error.stack || error);
  process.exitCode = 1;
} finally {
  if (chrome?.pid && chrome.exitCode === null && chrome.signalCode === null) {
    const exited = new Promise(resolve => chrome.once('exit', resolve));
    chrome.kill('SIGTERM');
    await exited;
  }
  if (server?.listening) await new Promise(resolve => server.close(resolve));
  await rm(temporary, { recursive: true, force: true });
}
