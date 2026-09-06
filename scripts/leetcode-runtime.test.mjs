import assert from "node:assert/strict";
import { test } from "node:test";
import { loadEmbeddedPyodide } from "./leetcode-runtime.mjs";

const base64 = (source) => Buffer.from(source).toString("base64");
const runtime = (loader) => ({
  runtime: {
    loader: base64(loader),
    asmModule: base64("export default () => 'embedded-module';"),
    wasm: base64(Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0)),
    stdlib: base64(Uint8Array.of(80, 75, 3, 4)),
  },
  lock: { packages: {} },
});

test("loads embedded modules and binaries with blob URLs unavailable", async () => {
  const originalSelf = globalThis.self;
  const originalCreateObjectURL = URL.createObjectURL;
  let networkCalls = 0;
  const nativeFetch = () => { networkCalls++; throw new Error("Network is disabled"); };
  globalThis.self = { fetch: nativeFetch };
  URL.createObjectURL = () => { throw new Error("Blob URLs are unavailable"); };
  try {
    const result = await loadEmbeddedPyodide(runtime(`
      export async function loadPyodide(options) {
        const wasm = await self.fetch(new Request(options.indexURL + 'pyodide.asm.wasm'));
        const stdlib = await self.fetch(new URL(options.stdLibURL));
        const repeated = await self.fetch(options.stdLibURL);
        return {
          module: options.createPyodideModule(),
          wasmType: wasm.headers.get('content-type'),
          stdlibType: stdlib.headers.get('content-type'),
          wasm: [...new Uint8Array(await wasm.arrayBuffer())],
          stdlib: [...new Uint8Array(await stdlib.arrayBuffer())],
          repeated: [...new Uint8Array(await repeated.arrayBuffer())],
          lock: options.lockFileContents,
        };
      }
    `));
    assert.deepEqual(result, {
      module: "embedded-module", wasmType: "application/wasm", stdlibType: "application/zip",
      wasm: [0, 97, 115, 109, 1, 0, 0, 0], stdlib: [80, 75, 3, 4], repeated: [80, 75, 3, 4], lock: { packages: {} },
    });
    assert.equal(networkCalls, 0);
    assert.equal(self.fetch, nativeFetch);
  } finally {
    globalThis.self = originalSelf;
    URL.createObjectURL = originalCreateObjectURL;
  }
});

test("restores fetch after initialization fails and permits a retry", async () => {
  const originalSelf = globalThis.self;
  const calls = [];
  const nativeFetch = function (input, options) {
    assert.equal(this, globalThis.self);
    calls.push([input, options]);
    return Promise.resolve(new Response("native response"));
  };
  globalThis.self = { fetch: nativeFetch };
  try {
    await assert.rejects(loadEmbeddedPyodide(runtime(`
      export async function loadPyodide() { throw new Error('initialization failed'); }
    `)), /initialization failed/);
    assert.equal(self.fetch, nativeFetch);
    const result = await loadEmbeddedPyodide(runtime(`
      export async function loadPyodide() {
        const response = await self.fetch('https://example.invalid/other', { method: 'GET' });
        return response.text();
      }
    `));
    assert.equal(result, "native response");
    assert.deepEqual(calls, [["https://example.invalid/other", { method: "GET" }]]);
    assert.equal(self.fetch, nativeFetch);
  } finally {
    globalThis.self = originalSelf;
  }
});
