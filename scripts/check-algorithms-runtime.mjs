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
const suiteStarted = Date.now();
// Keep the reported user solution independent of future reference-solution changes.
const moveZeroesSource = `class Solution:
    def moveZeroes(self, nums):
        write = 0
        for x in nums:
            if x != 0:
                nums[write] = x
                write += 1
        nums[write:] = [0] * (len(nums) - write)`;
const twoSumSource = `class Solution:
    def twoSum(self, nums, target):
        for left in range(len(nums)):
            for right in range(left + 1, len(nums)):
                if nums[left] + nums[right] == target:
                    return [left, right]`;

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
let nextWorkerMode = 'normal';
const workerFaultModes = new Map();
// Deterministic compatibility simulation, not a Safari/WebKit test: Blob URL
// imports and binary fetches are unavailable, even when the Worker created them.
function simulateWorkerCompatibility(mode) {
  const create = URL.createObjectURL.bind(URL);
  URL.createObjectURL = blob => {
    const url = create(blob);
    URL.revokeObjectURL(url);
    return url;
  };
  const nativeFetch = self.fetch.bind(self);
  self.fetch = (input, options) => {
    const url = typeof input === 'string' || input instanceof URL ? String(input) : input.url;
    if (url.startsWith('blob:')) return Promise.reject(new TypeError('regression: Worker Blob fetch unavailable'));
    return nativeFetch(input, options);
  };
  if (mode === 'normal') return;
  WebAssembly.instantiateStreaming = async () => {
    throw new WebAssembly.CompileError(mode === 'compile-failure'
      ? 'regression: unsupported WebAssembly module'
      : 'regression: streaming compilation unavailable');
  };
  if (mode !== 'compile-failure') return;
  // Permit the small feature probes used by Pyodide. Reject the actual Python
  // module whether the caller compiles first or instantiates its bytes directly.
  const isRuntimeModule = input => input instanceof WebAssembly.Module
    ? WebAssembly.Module.exports(input).length > 10
    : input?.byteLength > 1024;
  for (const method of ['compile', 'instantiate']) {
    const nativeMethod = WebAssembly[method].bind(WebAssembly);
    WebAssembly[method] = async (input, ...options) => {
      if (isRuntimeModule(input)) {
        throw new WebAssembly.CompileError('regression: unsupported WebAssembly module');
      }
      return nativeMethod(input, ...options);
    };
  }
}
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
      // Pause each real Worker before startup, then block HTTP(S) and Blob loads.
      const faultMode = nextWorkerMode;
      (async () => {
        if (targetInfo.type === 'worker') {
          workerSessions.add(sessionId);
          workerFaultModes.set(sessionId, faultMode);
          await cdp.send('Network.enable', {}, sessionId);
          await cdp.send('Network.setBlockedURLs', { urls: ['http://*', 'https://*'] }, sessionId);
          const expression = `(${simulateWorkerCompatibility.toString()})(${JSON.stringify(faultMode)})`;
          const injected = await cdp.send('Runtime.evaluate', { expression }, sessionId);
          if (injected.exceptionDetails) throw new Error('Could not inject Worker compatibility simulation');
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
    console.log(`PASS ${label}: Python startup; Worker HTTP(S) and all Blob loads blocked`);

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

    const twoSumResults = await evaluate(sessionId, `(async () => {
      const { code, tests, note, complexity, ...meta } = SOLUTIONS['two-sum'];
      return evaluate({ mode: 'core', userCode: ${JSON.stringify(twoSumSource)}, referenceCode: code, meta,
        cases: tests.map(value => ({ value })) }, 6000);
    })()`);
    assert.equal(twoSumResults.passed, true, `${label}: two-sum cases pass`);
    assert.deepEqual(twoSumResults.results.map(item => JSON.parse(item.actual)), [[0, 1], [1, 2], [0, 1], [0, 2]]);
    await evaluate(sessionId, `openProblem('two-sum'); elements.code_editor.value = ${JSON.stringify(twoSumSource)};
      elements.code_editor.dispatchEvent(new Event('input', { bubbles: true })); elements.submit_button.click();`);
    await waitFor(sessionId, `!evaluationInProgress && recordFor('two-sum').attempts === 1`);
    assert.equal(await evaluate(sessionId, `recordFor('two-sum').status === 'solved'
      && testConsoleState.evaluation?.passed && elements.result_panel.textContent.includes('全部通过')`), true);
    console.log(`PASS ${label}: two-sum evaluation and Submit button record successful progress`);

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

    await evaluate(sessionId, `(async () => { (await ensureJudge()).terminate(); judgeReady = null; })()`);
    nextWorkerMode = 'streaming-failure';
    const withoutStreaming = await evaluate(sessionId, `(async () => {
      await ensureJudge(); return (await evaluate(runtimeTestPayload, 6000)).passed;
    })()`);
    assert.equal(withoutStreaming, true, `${label}: Python works when streaming compilation is unavailable`);
    assert.equal(workerFaultModes.get([...workerSessions].at(-1)), 'streaming-failure');
    console.log(`PASS ${label}: Python startup and evaluation with streaming compilation unavailable`);

    await evaluate(sessionId, `(async () => { (await ensureJudge()).terminate(); judgeReady = null; })()`);
    nextWorkerMode = 'compile-failure';
    const compilationFailure = await evaluate(sessionId, `(async () => {
      let timer;
      const started = performance.now();
      const result = await Promise.race([
        ensureJudge().then(() => ({ failure: '' }), error => ({ failure: error.message })),
        new Promise(resolve => { timer = setTimeout(() => resolve({ failure: 'regression: compilation error was not reported promptly' }), 10000); }),
      ]);
      clearTimeout(timer);
      return { ...result, elapsed: performance.now() - started, cleared: judgeReady === null };
    })()`);
    assert.match(compilationFailure.failure, /regression: unsupported WebAssembly module/,
      `${label}: report the underlying compilation error instead of a startup timeout`);
    assert.ok(compilationFailure.elapsed < 10000, `${label}: compilation failure is reported promptly`);
    assert.equal(compilationFailure.cleared, true);
    assert.equal(workerFaultModes.get([...workerSessions].at(-1)), 'compile-failure');
    nextWorkerMode = 'normal';
    assert.equal(await evaluate(sessionId, `(async () => {
      await ensureJudge(); return (await evaluate(runtimeTestPayload, 6000)).passed;
    })()`), true, `${label}: real compilation failure permits a successful retry`);
    console.log(`PASS ${label}: underlying compilation failure reported in ${Math.round(compilationFailure.elapsed)}ms; fresh Worker retry succeeds`);

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
    assert.equal(workerSessions.size - workerCount, 6, `${label}: six real Worker lifetimes had HTTP(S) and Blob loads blocked`);
    assert.deepEqual(workerSetupErrors, []);
    assert.deepEqual(externalRequests, [], 'Runtime must not request external HTTP(S) resources');
    console.log(`PASS ${label}: infinite loop times out and a fresh Worker runs successfully`);
    await cdp.send('Target.disposeBrowserContext', { browserContextId });
  }
  console.log(`PASS all runtime checks (HTTP and standalone file; no external runtime requests; ${((Date.now() - suiteStarted) / 1000).toFixed(1)}s)`);
} catch (error) {
  console.error(error.stack || error);
  process.exitCode = 1;
} finally {
  if (chrome?.pid && chrome.exitCode === null && chrome.signalCode === null) {
    const exited = new Promise(resolve => chrome.once('exit', resolve));
    chrome.kill('SIGTERM');
    await exited;
  }
  // Crash reporting descendants can retain Chrome's pipes after the browser exits.
  for (const stream of chrome?.stdio || []) stream?.destroy?.();
  if (server?.listening) {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
  await rm(temporary, { recursive: true, force: true });
}
