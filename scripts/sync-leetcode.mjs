import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { gzipSync, gunzipSync } from "node:zlib";

// Update the revision and checksum together when adopting a new upstream build.
const revision = "7227b9f9c3d000b8dcb70c80e2d092146996d280";
const checksum = "080967140ea291fba082766122a8c55cd632ed1d897aa83c1c1ea4c9a4cdc75f";
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
if (application.split(marker).length !== 2 || !application.includes('class="hero-actions"')) {
  throw new Error("上游启动结构已改变，请检查站点适配逻辑");
}
const adapter = await readFile(new URL("./leetcode-site.js", import.meta.url), "utf8");
const adapted = application.replace(marker, () => `${adapter}\n${marker}`);
const compressed = gzipSync(Buffer.from(adapted), { level: 9 }).toString("base64");
const output = shell.replace(payload[0], () => `const COMPRESSED_APP = "${compressed}";`)
  .replace("<head>", `<head>\n  <!-- Source: ${upstream}\n       SHA-256: ${checksum}\n       Rebuilt with scripts/sync-leetcode.mjs; site changes: scripts/leetcode-site.js -->`);
await writeFile(new URL("../leetcode.html", import.meta.url), output);
console.log(`已更新 leetcode.html：${(Buffer.byteLength(output) / 1024 / 1024).toFixed(2)} MiB，来源 ${revision.slice(0, 7)}`);
