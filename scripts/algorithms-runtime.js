// Inlined by sync-algorithms.mjs; keep the downloaded HTML self-contained.
// src/ui/judge.js
const JUDGE_WORKER_SOURCE = `
let pyodide = null;
let harness = "";
let formatterBundle = null;
let formatterScript = "";
// A data: module stack can contain megabytes of source; report its message instead.
const errorText = (error) => String(error && error.message || error);
const progress = (stage) => self.postMessage({ type: "init-progress", stage });

self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === "init") {
    let stage = "读取内嵌资源";
    try {
      progress(stage);
      harness = message.harness;
      formatterBundle = message.formatterBundle;
      formatterScript = message.formatterScript;
      const wasmBytes = Uint8Array.fromBase64(message.runtime.wasm);
      const stdlibBytes = Uint8Array.fromBase64(message.runtime.stdlib);
      const nativeFetch = self.fetch.bind(self);
      const nativeInstantiateStreaming = WebAssembly.instantiateStreaming;
      let rejectRuntime;
      const runtimeFailure = new Promise((_, reject) => { rejectRuntime = reject; });
      // Pyodide logs some initialization failures without rejecting loadPyodide.
      const unhandledFailure = (event) => {
        event.preventDefault();
        rejectRuntime(event.reason);
      };
      self.addEventListener("unhandledrejection", unhandledFailure);
      try {
        // Bytes are already in memory. Avoid a second fetch through an opaque-origin Blob URL.
        self.fetch = (url, options) => {
          const address = url instanceof Request ? url.url : String(url);
          if (address === "https://offline.local/pyodide.asm.wasm") {
            return Promise.resolve(new Response(wasmBytes, { headers: { "Content-Type": "application/wasm" } }));
          }
          if (address === "https://offline.local/python_stdlib.zip") {
            return Promise.resolve(new Response(stdlibBytes, { headers: { "Content-Type": "application/zip" } }));
          }
          return nativeFetch(url, options);
        };
        // Preserve Pyodide's import hooks, but use buffered compilation for embedded bytes.
        // Relay failures because its instantiateWasm hook otherwise only logs and hangs.
        WebAssembly.instantiateStreaming = async (response, imports) => {
          stage = "编译 Python WebAssembly";
          progress(stage);
          try {
            const result = await WebAssembly.instantiate(await (await response).arrayBuffer(), imports);
            stage = "初始化 Python 标准库";
            progress(stage);
            return result;
          } catch (error) {
            rejectRuntime(error);
            throw error;
          }
        };
        // Inline modules also load in opaque-origin workers used by offline files.
        stage = "加载 Python 模块";
        progress(stage);
        const [loader, asmModule] = await Promise.all([
          import("data:text/javascript;base64," + message.runtime.loader),
          import("data:text/javascript;base64," + message.runtime.asmModule),
        ]);
        pyodide = await Promise.race([loader.loadPyodide({
          indexURL: "https://offline.local/",
          stdLibURL: "https://offline.local/python_stdlib.zip",
          lockFileContents: message.lock,
          packageBaseUrl: "https://offline.local/",
          createPyodideModule: asmModule.default,
        }), runtimeFailure]);
      } finally {
        self.fetch = nativeFetch;
        WebAssembly.instantiateStreaming = nativeInstantiateStreaming;
        self.removeEventListener("unhandledrejection", unhandledFailure);
      }
      self.postMessage({ type: "ready" });
    } catch (error) {
      self.postMessage({ type: "init-error", error: stage + "失败：" + errorText(error) });
    }
    return;
  }
  if (message.type !== "run" && message.type !== "format") return;
  try {
    let result;
    if (message.type === "run") {
      pyodide.globals.set("payload_json", JSON.stringify(message.payload));
      result = await pyodide.runPythonAsync(harness + "\\nRESULT_JSON");
    } else {
      if (formatterBundle) {
        pyodide.globals.set("formatter_bundle_json", JSON.stringify(formatterBundle));
        pyodide.globals.set("formatter_line_length", formatterBundle.lineLength);
        formatterBundle = null;
      }
      pyodide.globals.set("formatter_source", message.source);
      result = await pyodide.runPythonAsync(formatterScript + "\\nFORMAT_RESULT_JSON");
    }
    self.postMessage({ type: message.type + "-result", id: message.id, result: JSON.parse(result) });
  } catch (error) {
    self.postMessage({ type: message.type + "-error", id: message.id, error: errorText(error) });
  }
};
`;

let judgeReady = null;
let pendingPythonRequest = null;
let pythonRequestSequence = 0;

function setRuntimeStatus(status, message) {
  elements.runtime_status.className = `runtime-status ${status}`;
  elements.runtime_status.querySelector("span:last-child").textContent = message;
  elements.runtime_status.title = message;
  elements.runtime_status.setAttribute("aria-label", message);
}

function startWorkerJudge() {
  return new Promise((resolve, reject) => {
    let worker = null;
    let initialized = false;
    let stopped = false;
    let initTimer = null;
    let stage = "启动 Python Worker";
    const fail = (error) => {
      if (stopped) return;
      stopped = true;
      if (initTimer) clearTimeout(initTimer);
      if (worker) worker.terminate();
      if (!initialized) {
        initialized = true;
        reject(error);
      } else {
        if (pendingPythonRequest) { pendingPythonRequest.reject(error); pendingPythonRequest = null; }
        judgeReady = null;
        setRuntimeStatus("", "Python 评测进程已停止，将在下次运行时重启");
      }
    };
    try {
      // A data URL can start a module worker from file:// without a cross-origin Blob fetch.
      const workerUrl = `data:text/javascript;base64,${new TextEncoder().encode(JUDGE_WORKER_SOURCE).toBase64()}`;
      worker = new Worker(workerUrl, { type: "module", name: "algorithm-python-judge" });
      initTimer = setTimeout(() => fail(new Error(stage + "超时，请重试；若持续失败，请检查浏览器是否允许 WebAssembly")), 20000);
      worker.onmessage = (event) => {
        if (stopped) return;
        const message = event.data;
        if (message.type === "init-progress" && !initialized) {
          stage = message.stage;
          setRuntimeStatus("loading", stage + "…");
        } else if (message.type === "ready") {
          if (initialized) return;
          initialized = true;
          clearTimeout(initTimer);
          setRuntimeStatus("ready", "离线 Python 已就绪");
          resolve(worker);
        } else if (message.type === "init-error") {
          fail(new Error(message.error));
        } else if (pendingPythonRequest?.id === message.id && message.type === `${pendingPythonRequest.type}-result`) {
          const pending = pendingPythonRequest; pendingPythonRequest = null; pending.resolve(message.result);
        } else if (pendingPythonRequest?.id === message.id && message.type === `${pendingPythonRequest.type}-error`) {
          const pending = pendingPythonRequest; pendingPythonRequest = null; pending.reject(new Error(message.error));
        }
      };
      worker.onerror = (event) => {
        event.preventDefault();
        const location = event.filename && !event.filename.startsWith("data:") ? ` (${event.filename}:${event.lineno || 0})` : "";
        fail(new Error(`${event.message || "Python Worker 发生错误"}${location}`));
      };
      worker.onmessageerror = () => fail(new Error(stage + "失败：无法读取 Worker 消息"));
      worker.postMessage({
        type: "init",
        runtime: RUNTIME_BASE64,
        lock: PYODIDE_LOCK,
        harness: PYTHON_HARNESS,
        formatterBundle: FORMATTER_BUNDLE,
        formatterScript: PYTHON_FORMATTER,
      });
    } catch (error) {
      fail(error);
    }
  });
}

function ensureJudge() {
  if (judgeReady) return judgeReady;
  setRuntimeStatus("loading", "正在启动内嵌 Python 运行时…");
  judgeReady = startWorkerJudge()
    .catch((error) => {
      judgeReady = null;
      setRuntimeStatus("", "Python 启动失败，可点击运行重试");
      elements.runtime_status.title = error.message;
      throw error;
    });
  return judgeReady;
}

async function evaluate(payload, timeoutMs) {
  return requestPythonWorker("run", { payload }, timeoutMs);
}

async function formatPythonSource(source, timeoutMs = 12000) {
  return requestPythonWorker("format", { source }, timeoutMs);
}

async function requestPythonWorker(type, payload, timeoutMs) {
  const worker = await ensureJudge();
  if (pendingPythonRequest) throw new Error("已有 Python 任务正在运行");
  const id = ++pythonRequestSequence;
  const action = type === "format" ? "格式化" : "运行";
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pendingPythonRequest = null;
      worker.terminate();
      judgeReady = null;
      setRuntimeStatus("", `上次${action}超时，运行时将在下次操作时重启`);
      reject(new Error(`${action}超过 ${Math.round(timeoutMs / 1000)} 秒，已终止 Python 进程`));
    }, timeoutMs);
    pendingPythonRequest = {
      id,
      type,
      resolve: (value) => { clearTimeout(timer); resolve(value); },
      reject: (error) => { clearTimeout(timer); reject(error); },
    };
    try {
      worker.postMessage({ type, id, ...payload });
    } catch (error) {
      const pending = pendingPythonRequest;
      pendingPythonRequest = null;
      pending.reject(error);
    }
  });
}

function prewarmJudge() {
  setTimeout(() => ensureJudge().catch((error) => console.warn("离线 Python 预热失败：", error)), 500);
}
