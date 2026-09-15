# Algorithm & AI Study Hub

原生 HTML/CSS/JavaScript 静态学习站，包含 100 道 Python 算法练习和 Transformer、RL、SFT、Agent 面试题。无需安装依赖或构建即可运行。

## 项目结构

```text
.
├── index.html                 # Study Hub 首页
├── pages/
│   ├── algorithms.html        # 算法训练场独立离线页面
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
│       ├── theme.js           # 全站深浅模式与偏好同步
│       └── interview.js       # 面试页渲染与交互
├── data/
│   └── interviews/
│       ├── transformer.js
│       ├── rl.js
│       ├── sft.js
│       └── agent.js
├── scripts/
│   ├── sync-algorithms.mjs    # 校验、同步及调整站点命名
│   ├── algorithms-site.js    # 注入返回 Study Hub 导航
│   ├── algorithms-runtime.js # 内嵌 Python 启动与错误处理适配
│   └── check-algorithms-runtime.mjs # 浏览器运行时回归检查
└── NOTICE.md                 # 原始来源与许可说明
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

## 深浅模式

首页、四个面试页及算法训练场均可通过右上角的太阳 / 月亮按钮切换深浅模式。同一站点来源下共用主题偏好，刷新、跳转和已打开的标签页保持同步。首次使用跟随系统；已有算法页主题会迁移为全站偏好，手动选择后优先使用所选模式。本地存储不可用时，当前页面仍可切换。

共享逻辑位于 `assets/js/theme.js`，使用独立的 `study-hub:theme` 存储键。算法页的旧设置与备份格式继续兼容；恢复备份中的主题也会应用到全站。算法页内嵌这份脚本以保留独立离线使用，修改后需运行下面的同步命令重新生成。直接通过 `file://` 打开时，跨文件共享偏好取决于浏览器的本地文件存储策略。

## 算法训练场

`pages/algorithms.html` 基于未压缩单文件版本，约 20 MiB，直接保留 HTML、CSS 和 JavaScript，内嵌 100 道题、题解、评测器、格式化器和 Python 运行时。首次访问需下载完整页面，保存文件后可直接离线打开；使用支持原生 Base64 的现代浏览器。

站点统一使用「算法训练场」名称，并在顶部添加返回 Study Hub 首页（`../index.html`）的入口，窄屏时独占一行。题目与评测逻辑沿用源文件，Python 加载增加浏览器兼容适配；历史存储键和 JSON 备份格式保持兼容，已有代码、笔记和进度在同一站点来源下继续可用。

Python Worker 通过内嵌 `data:` 地址导入 JavaScript 模块，Wasm 和标准库直接从内存提供，不再依赖 Worker 内的 Blob URL 读取。Wasm 使用已缓存字节进行编译，并保留 Pyodide 的初始化钩子；底层编译失败会立即传回页面，避免被加载器吞掉后只显示“启动超时”。状态栏显示当前启动阶段，失败后可点击运行重试。运行代码和格式化无需联网。

题目示例保留原始内容，来源和许可集中在 [NOTICE.md](NOTICE.md)。

### 更新离线页面

当前固定上游提交 `5828bd249feede51f752ee42f9ee92254ac84933`，源文件的 SHA-256 为 `a745776bc832a1080ebfa1ad6f266fa39aaf5d2e01405e7ef55803570d7f9870`。

下载方式见 [来源说明](NOTICE.md)，也可直接使用本地原始文件：

```bash
node scripts/sync-algorithms.mjs /path/to/lc_offline.html
```

同步脚本校验来源文件，替换站点展示名称，加入 `scripts/algorithms-site.js` 中的返回导航，并内嵌共享主题脚本和 `scripts/algorithms-runtime.js` 中的启动适配，直接写入 `pages/algorithms.html`，不压缩或添加解压启动页。题库、编辑器、学习记录存储和 Python 评测逻辑保留源文件实现，运行时不依赖外部脚本。修改运行时适配后也需重新运行同步命令。升级上游版本时先检查文案、导航、主题和运行时适配点，再一起更新提交号与校验值。

### 运行时回归检查

安装 Node.js 和 Chrome 后运行（无需 npm 依赖）：

```bash
node scripts/check-algorithms-runtime.mjs
```

脚本使用临时浏览器配置，检查 HTTP 和独立 `file://` 页面的 Python 启动、判题、格式化、启动失败重试及运行超时后恢复。测试禁止 Worker 请求外部 HTTP(S) 资源及读取 Blob URL，并模拟流式编译不可用、Wasm 编译失败，验证兼容加载和底层错误上报。默认使用 macOS Chrome 路径，其他安装位置可通过 `CHROME_PATH` 指定；这些兼容性模拟不代表实际 Safari 测试。
