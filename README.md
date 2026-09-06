# Algorithm & AI Study Hub

原生 HTML/CSS/JavaScript 静态学习站，包含 LeetCode Hot 100 和 Transformer、RL、SFT、Agent 面试题。无需安装依赖或构建即可运行。

## LeetCode 离线训练场

`leetcode.html` 基于 [leetcode_hot100_html 的压缩单文件版本](https://github.com/yqstar/leetcode_hot100_html/blob/7227b9f9c3d000b8dcb70c80e2d092146996d280/lc_offline_compact.html)，约 12 MiB，内嵌 100 道题、题解、评测器、格式化器和 Python 运行时。首次访问需下载完整页面，保存文件后可直接离线打开；使用支持原生 Base64 和 DecompressionStream 的现代浏览器。

保留原有 `leetcode.html` 地址，并添加返回 Study Hub 的入口。首次使用新版且没有 `lc-offline:v1` 数据时，会将同一浏览器、同一站点的旧版 `lc_done` 完成标记迁移为已掌握；不会覆盖已有新版记录，也不会删除旧数据。代码、笔记和进度保存在浏览器，可通过页面导出 JSON 备份。

上游题面、Pyodide、CPython 和 Black 的来源及许可说明保留在页面内，详见[上游说明](https://github.com/yqstar/leetcode_hot100_html#readme)。其他面试题页面继续使用 `assets/interview.js` 与 `assets/common.css`。

### 更新离线页面

当前固定上游提交 `7227b9f9c3d000b8dcb70c80e2d092146996d280`，下载文件的 SHA-256 为 `080967140ea291fba082766122a8c55cd632ed1d897aa83c1c1ea4c9a4cdc75f`。

```bash
curl -sS -L --fail https://raw.githubusercontent.com/yqstar/leetcode_hot100_html/7227b9f9c3d000b8dcb70c80e2d092146996d280/lc_offline_compact.html -o /tmp/lc_offline_compact.html
node scripts/sync-leetcode.mjs /tmp/lc_offline_compact.html
```

同步脚本校验来源文件，解压后加入 `scripts/leetcode-site.js` 中的站点导航和旧进度迁移，并通过 `scripts/leetcode-runtime.mjs` 适配 Python 加载：JavaScript 模块使用内嵌 data URL，Wasm 和标准库直接从内存读取，避免 Worker 中的 `blob:null` 模块导入失败。生成的 `leetcode.html` 仍是独立文件，运行时不依赖外部脚本。升级上游版本时先检查适配点，再一起更新脚本中的提交号与校验值；题库和评测器继续在上游项目维护。

运行时适配的回归测试（使用支持原生 Base64 的 Node.js）：

```bash
node --test scripts/leetcode-runtime.test.mjs
```
