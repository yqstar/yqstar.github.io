// This function is embedded into the offline page's module Worker at sync time.
// Keep it self-contained: the generated HTML must also run without this file.
export async function loadEmbeddedPyodide(message) {
  const baseUrl = "https://offline.local/";
  const binaries = new Map([
    [baseUrl + "pyodide.asm.wasm", [message.runtime.wasm, "application/wasm"]],
    [baseUrl + "python_stdlib.zip", [message.runtime.stdlib, "application/zip"]],
  ]);
  const nativeFetch = self.fetch;
  try {
    // A data: Worker may have an opaque origin. Its blob:null module URLs are
    // not reliably importable across browsers, so import the embedded JS as data:.
    self.fetch = async (input, options) => {
      const url = typeof input === "string" ? input : input.url || String(input);
      const binary = binaries.get(url);
      if (!binary) return nativeFetch.call(self, input, options);
      return new Response(Uint8Array.fromBase64(binary[0]), {
        headers: { "Content-Type": binary[1] },
      });
    };
    const [loader, asmModule] = await Promise.all([
      import("data:text/javascript;base64," + message.runtime.loader),
      import("data:text/javascript;base64," + message.runtime.asmModule),
    ]);
    // Preserve Pyodide's own Wasm initialization hook. Only its binary reads
    // are supplied from memory; neither resource needs a network or blob fetch.
    return await loader.loadPyodide({
      indexURL: baseUrl,
      stdLibURL: baseUrl + "python_stdlib.zip",
      lockFileContents: message.lock,
      packageBaseUrl: baseUrl,
      createPyodideModule: asmModule.default,
    });
  } finally {
    self.fetch = nativeFetch;
  }
}

function replaceOnce(source, before, after, label) {
  if (source.split(before).length !== 2) {
    throw new Error(`上游 Python ${label}结构已改变，请检查离线运行时适配逻辑。`);
  }
  return source.replace(before, () => after);
}

export function patchRuntimeLoader(application) {
  const legacyHelper = `function createRuntimeUrl(base64, type) {
  return URL.createObjectURL(new Blob([Uint8Array.fromBase64(base64)], { type }));
}`;
  const legacyInitialization = `      const nativeFetch = self.fetch.bind(self);
      const runtimeUrls = [];
      try {
        runtimeUrls.push(
          createRuntimeUrl(message.runtime.loader, "text/javascript"),
          createRuntimeUrl(message.runtime.asmModule, "text/javascript"),
          createRuntimeUrl(message.runtime.wasm, "application/wasm"),
          createRuntimeUrl(message.runtime.stdlib, "application/zip"),
        );
        // Preserve Pyodide's Wasm initialization hook while serving the embedded binary.
        self.fetch = (url, options) => nativeFetch(
          String(url) === "https://offline.local/pyodide.asm.wasm" ? runtimeUrls[2] : url, options,
        );
        const [loader, asmModule] = await Promise.all([import(runtimeUrls[0]), import(runtimeUrls[1])]);
        pyodide = await loader.loadPyodide({
          indexURL: "https://offline.local/",
          stdLibURL: runtimeUrls[3],
          lockFileContents: message.lock,
          packageBaseUrl: "https://offline.local/",
          createPyodideModule: asmModule.default,
        });
      } finally {
        self.fetch = nativeFetch;
        runtimeUrls.forEach((url) => URL.revokeObjectURL(url));
      }`;
  let patched = replaceOnce(application, legacyHelper, loadEmbeddedPyodide.toString(), "加载函数");
  patched = replaceOnce(patched, legacyInitialization, "      pyodide = await loadEmbeddedPyodide(message);", "初始化");
  return patched;
}
