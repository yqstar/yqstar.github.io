# Algorithm & AI Study Hub

原生 HTML/CSS/JavaScript 静态学习站，包含 LeetCode Hot 100 和 Transformer、RL、SFT、Agent 面试题。无需安装依赖或构建即可运行。

## 项目结构

```text
.
├── index.html                 # Study Hub 首页
├── pages/
│   ├── leetcode.html          # LeetCode 独立离线页面
│   └── interviews/
│       ├── transformer-interview.html
│       ├── rl-interview.html
│       ├── sft-interview.html
│       └── agent-interview.html
├── assets/
│   ├── css/
│   │   ├── site.css           # 站点共用基础样式
│   │   ├── home.css           # 首页样式
│   │   └── interview.css      # 面试页样式
│   └── js/
│       └── interview.js       # 面试页渲染与交互
├── data/
│   └── interviews/
│       ├── transformer.js
│       ├── rl.js
│       ├── sft.js
│       └── agent.js
└── scripts/
    ├── sync-leetcode.mjs      # 校验并同步 LeetCode 源文件
    └── leetcode-site.js       # 注入返回 Study Hub 导航
```

根目录只保留首页 `index.html`，子页面归入 `pages/`，其中四个面试专题统一放在 `pages/interviews/`。样式、交互和面试题库分别放在 `assets/css/`、`assets/js/` 和 `data/interviews/`。站内导航均指向新路径；面试页文件名不变，同一站点的浏览器题库继续沿用原存储键。

## 本地预览

在项目根目录运行：

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8000/>。不需要 npm 或构建步骤，也可以直接用浏览器打开 `index.html`。

## 维护面试题

编辑 `data/interviews/` 下对应主题的 JavaScript 文件，修改其中的 `const DATA` 对象。面试页通过普通 `<script>` 加载题库与 `assets/js/interview.js`，再调用 `initInterview(DATA)`，支持直接通过 `file://` 打开。

调整共用外观时编辑 `assets/css/site.css`，面试页专属样式编辑 `assets/css/interview.css`。页面内的导入、导出和本地存储用于浏览器中的学习数据，不会写回仓库文件；需要更新站点题库时仍应修改对应的 `data/interviews/*.js`。

## LeetCode 离线训练场

`pages/leetcode.html` 基于 [leetcode_hot100_html 的压缩单文件版本](https://github.com/yqstar/leetcode_hot100_html/blob/5828bd249feede51f752ee42f9ee92254ac84933/lc_offline_compact.html)，约 12 MiB，内嵌 100 道题、题解、评测器、格式化器和 Python 运行时。首次访问需下载完整页面，保存文件后可直接离线打开；使用支持原生 Base64 和 DecompressionStream 的现代浏览器。

`pages/leetcode.html` 与源文件保持相同实现，唯一新增功能是顶部返回 Study Hub 首页（`../index.html`）的入口，窄屏时独占一行。代码、笔记和进度沿用源文件的浏览器存储逻辑，可通过页面导出 JSON 备份。

上游题面、Pyodide、CPython 和 Black 的来源及许可说明保留在页面内，详见[上游说明](https://github.com/yqstar/leetcode_hot100_html#readme)。

### 更新离线页面

当前固定上游提交 `5828bd249feede51f752ee42f9ee92254ac84933`，源文件的 SHA-256 为 `465df34ec046883997439d213bb2ea4d4a2d1f19d861cc374c251bba37cf0ae1`。

```bash
curl -sS -L --fail https://raw.githubusercontent.com/yqstar/leetcode_hot100_html/5828bd249feede51f752ee42f9ee92254ac84933/lc_offline_compact.html -o /tmp/lc_offline_compact.html
node scripts/sync-leetcode.mjs /tmp/lc_offline_compact.html
```

也可直接使用本地源文件：

```bash
node scripts/sync-leetcode.mjs /Users/yq/Documents/leetcode_hot100_html/lc_offline_compact.html
```

同步脚本校验来源文件，解压后仅加入 `scripts/leetcode-site.js` 中的返回导航，再重新压缩并写入 `pages/leetcode.html`。题库、界面、编辑器、存储、评测器及 Python 加载均保留源文件实现，运行时不依赖外部脚本。升级上游版本时先检查导航注入点，再一起更新脚本中的提交号与校验值。
