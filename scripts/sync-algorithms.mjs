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

// data: workers have opaque origins. Import embedded JS as data: modules to
// avoid browser-dependent CORS failures when importing their blob:null URLs.
// Keep binary Blob URLs and Pyodide's Wasm initialization hook unchanged.
const runtimeChanges = [
  ['          createRuntimeUrl(message.runtime.loader, "text/javascript"),\n          createRuntimeUrl(message.runtime.asmModule, "text/javascript"),\n', ''],
  ['? runtimeUrls[2] : url, options,', '? runtimeUrls[0] : url, options,'],
  ['        const [loader, asmModule] = await Promise.all([import(runtimeUrls[0]), import(runtimeUrls[1])]);',
    '        // Inline modules also load in opaque-origin workers used by offline files.\n        const [loader, asmModule] = await Promise.all([\n          import("data:text/javascript;base64," + message.runtime.loader),\n          import("data:text/javascript;base64," + message.runtime.asmModule),\n        ]);'],
  ['          stdLibURL: runtimeUrls[3],', '          stdLibURL: runtimeUrls[1],'],
];
for (const [before, after] of runtimeChanges) {
  if (application.split(before).length !== 2) {
    throw new Error(`上游运行时逻辑已改变，请检查离线模块加载适配：${before}`);
  }
  application = application.replace(before, () => after);
}

// Route startup, manual changes and backup restoration through the same preference.
// Keep the upstream settings shape intact for existing records and JSON backups.
const themeChanges = [
  ['function applyTheme(theme) {', 'function applyTheme(theme, persist = true) {'],
  ['  elements.theme_button.title = label;\n}',
    '  elements.theme_button.title = label;\n  window.StudyHubTheme.set(theme, { persist });\n}'],
  ['  showCatalog();\n  applyTheme(state.settings.theme);',
    '  showCatalog();\n  applyTheme(window.StudyHubTheme.get(), false);'],
  ['    lastImportUndo = { records: plan.beforeRecords, settings: plan.beforeSettings };',
    '    lastImportUndo = { records: plan.beforeRecords, settings: plan.beforeSettings, themePreference: plan.settingsChanged ? window.StudyHubTheme.getPreference() : null };\n    if (plan.settingsChanged) window.StudyHubTheme.setPreference(state.settings.theme, { persist: !memoryOnly });'],
  ['    commitImportedState(next, !storageAvailable);\n    lastImportUndo = null;',
    '    commitImportedState(next, !storageAvailable);\n    if (lastImportUndo.themePreference) window.StudyHubTheme.setPreference(lastImportUndo.themePreference, { persist: storageAvailable });\n    lastImportUndo = null;'],
  ['initializeCatalogFilters();\napplyTheme(state.settings.theme);',
    'window.addEventListener("study-hub-theme-change", event => applyTheme(event.detail.theme, false));\n\ninitializeCatalogFilters();\napplyTheme(state.settings.theme, false);'],
];
for (const [before, after] of themeChanges) {
  if (application.split(before).length !== 2) {
    throw new Error(`上游主题逻辑已改变，请检查全站主题适配：${before}`);
  }
  application = application.replace(before, () => after);
}
const marker = "const state = loadState();";
if (application.split(marker).length !== 2 || !application.includes('id="app-header"') || !application.includes('class="header-leading"')) {
  throw new Error("上游启动结构已改变，请检查站点适配逻辑");
}
const adapter = await readFile(new URL("./algorithms-site.js", import.meta.url), "utf8");
const theme = await readFile(new URL("../assets/js/theme.js", import.meta.url), "utf8");
if (/<\/script/i.test(theme)) throw new Error("共享主题脚本包含 script 结束标记，无法安全内嵌。");
const adapted = application.replace(marker, () => `${adapter}\n${marker}\nstate.settings.theme = window.StudyHubTheme.get();`);
const output = adapted
  .replace('<meta name="color-scheme" content="light dark">', () => `<meta name="color-scheme" content="light dark">\n<script>\n${theme}</script>`)
  .replace("<head>", `<head>\n  <!-- Source revision: ${revision}; see ../NOTICE.md\n       SHA-256: ${checksum}\n       Rebuilt with scripts/sync-algorithms.mjs; site changes: display naming, navigation, shared appearance and offline module loading -->`);
await mkdir(new URL("../pages/", import.meta.url), { recursive: true });
await writeFile(new URL("../pages/algorithms.html", import.meta.url), output);
console.log(`已更新 pages/algorithms.html：${(Buffer.byteLength(output) / 1024 / 1024).toFixed(2)} MiB，来源 ${revision.slice(0, 7)}`);
