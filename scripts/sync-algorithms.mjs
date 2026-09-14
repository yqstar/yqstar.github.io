import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";

// Update the revision and checksum together when adopting a new upstream build.
const revision = "5828bd249feede51f752ee42f9ee92254ac84933";
const checksum = "a745776bc832a1080ebfa1ad6f266fa39aaf5d2e01405e7ef55803570d7f9870";
const sourcePath = process.argv[2];
if (!sourcePath) throw new Error("用法：node scripts/sync-algorithms.mjs /path/to/lc_offline.html");

const source = await readFile(sourcePath);
if (createHash("sha256").update(source).digest("hex") !== checksum) {
  throw new Error("上游文件 SHA-256 不符；升级前请核实版本并更新 revision 和 checksum。");
}

let application = source.toString("utf8");
// Change presentation labels only. Keep question data, storage and backup formats compatible.
const labels = [
  ['<title>Python 离线训练场</title>', '<title>Python 算法训练场</title>'],
  ['完全离线的 LC Python 学习与评测工具', '完全离线的 Python 算法学习与评测工具'],
  ['<div class="brand-title">Python 离线训练场</div>', '<div class="brand-title">Python 算法训练场</div>'],
  ['LC 算法题 · 题面、评测、题解与笔记', '算法练习 · 题面、评测、题解与笔记'],
  ['HOT 100 · PYTHON', 'ALGORITHMS · PYTHON'],
  ['题面版权归 LC 及原作者', '题面及图片版权归原作者'],
  ['题面来自力扣 LC 题单的离线快照，仅用于个人学习。网页中的本地评测是练习辅助，不等同于力扣官方判题。',
    '题面为公开题单的离线快照，仅用于个人学习。本地评测用于辅助练习。<a href="../NOTICE.md" style="color:inherit">来源与许可</a>'],
  ['"# LC 学习记录"', '"# 算法学习记录"'],
  ['由 Python 离线训练场导出于', '由 Python 算法训练场导出于'],
  ['`lc-progress-${stamp}.${extension}`', '`algorithm-progress-${stamp}.${extension}`'],
  ['name: "lc-python-judge"', 'name: "algorithm-python-judge"'],
];
for (const [before, after] of labels) {
  if (application.split(before).length !== 2) {
    throw new Error(`上游文案已改变，请检查站点命名适配：${before}`);
  }
  application = application.replace(before, () => after);
}
const marker = "const state = loadState();";
if (application.split(marker).length !== 2 || !application.includes('id="app-header"') || !application.includes('class="header-leading"')) {
  throw new Error("上游启动结构已改变，请检查站点适配逻辑");
}
const adapter = await readFile(new URL("./algorithms-site.js", import.meta.url), "utf8");
const adapted = application.replace(marker, () => `${adapter}\n${marker}`);
const output = adapted
  .replace("<head>", `<head>\n  <!-- Source revision: ${revision}; see ../NOTICE.md\n       SHA-256: ${checksum}\n       Rebuilt with scripts/sync-algorithms.mjs; site changes: display naming and scripts/algorithms-site.js navigation -->`);
await mkdir(new URL("../pages/", import.meta.url), { recursive: true });
await writeFile(new URL("../pages/algorithms.html", import.meta.url), output);
console.log(`已更新 pages/algorithms.html：${(Buffer.byteLength(output) / 1024 / 1024).toFixed(2)} MiB，来源 ${revision.slice(0, 7)}`);
