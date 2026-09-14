import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { gzipSync, gunzipSync } from "node:zlib";

// Update the revision and checksum together when adopting a new upstream build.
const revision = "5828bd249feede51f752ee42f9ee92254ac84933";
const checksum = "465df34ec046883997439d213bb2ea4d4a2d1f19d861cc374c251bba37cf0ae1";
const upstream = `https://github.com/yqstar/leetcode_hot100_html/blob/${revision}/lc_offline_compact.html`;
const sourcePath = process.argv[2];
if (!sourcePath) throw new Error("用法：node scripts/sync-leetcode.mjs /path/to/lc_offline_compact.html");

const source = await readFile(sourcePath);
if (createHash("sha256").update(source).digest("hex") !== checksum) {
  throw new Error("上游文件 SHA-256 不符；升级前请核实版本并更新 revision 和 checksum。");
}

const shell = source.toString("utf8");
const payload = shell.match(/const COMPRESSED_APP = "([A-Za-z0-9+/=]+)";/);
if (!payload) throw new Error("未找到上游压缩内容");
const application = gunzipSync(Buffer.from(payload[1], "base64")).toString("utf8");
const marker = "const state = loadState();";
if (application.split(marker).length !== 2 || !application.includes('id="app-header"') || !application.includes('class="header-leading"')) {
  throw new Error("上游启动结构已改变，请检查站点适配逻辑");
}
const adapter = await readFile(new URL("./leetcode-site.js", import.meta.url), "utf8");
const adapted = application.replace(marker, () => `${adapter}\n${marker}`);
const compressed = gzipSync(Buffer.from(adapted), { level: 9 }).toString("base64");
const output = shell.replace(payload[0], () => `const COMPRESSED_APP = "${compressed}";`)
  .replace("<head>", `<head>\n  <!-- Source: ${upstream}\n       SHA-256: ${checksum}\n       Rebuilt with scripts/sync-leetcode.mjs; site changes: scripts/leetcode-site.js (Study Hub navigation only) -->`);
await mkdir(new URL("../pages/", import.meta.url), { recursive: true });
await writeFile(new URL("../pages/leetcode.html", import.meta.url), output);
console.log(`已更新 pages/leetcode.html：${(Buffer.byteLength(output) / 1024 / 1024).toFixed(2)} MiB，来源 ${revision.slice(0, 7)}`);
