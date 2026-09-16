# Algorithm & AI Study Hub

原生 HTML/CSS/JavaScript 静态学习站，包含 100 道 Python 算法练习、Transformer、RL、SFT、Agent 的 114 道精选面试题，以及 39 篇中文论文全文和 8 篇经典 AI 论文解析。无需安装依赖或构建即可运行。

## 项目结构

```text
.
├── index.html                 # Study Hub 首页
├── pages/
│   ├── algorithms.html        # 算法训练场独立离线页面
│   ├── interviews.html        # AI 专题目录，四个主题的独立入口
│   ├── papers.html            # 论文阅读专题
│   └── interviews/
│       ├── transformer-interview.html
│       ├── rl-interview.html
│       ├── sft-interview.html
│       └── agent-interview.html
├── assets/
│   ├── papers/zh/             # 39 份已完成的中文 PDF，点击后按需打开
│   ├── css/
│   │   ├── site.css           # 站点共用基础样式
│   │   ├── home.css           # 首页样式
│   │   ├── interview.css      # 面试页样式
│   │   └── papers.css         # 论文阅读页样式
│   └── js/
│       ├── navigation.js      # 全站共用的四入口顶部导航（算法页内嵌）
│       ├── theme.js           # 全站深浅模式与偏好同步
│       ├── interview.js       # 面试页渲染与交互
│       └── papers.js          # 论文筛选、阅读进度、笔记与备份
├── data/
│   ├── papers.js              # 论文资料、来源与阅读导引
│   ├── translated-papers.js   # 中文译文书架、摘要、导引与英文原文链接
│   ├── paper-translations-manifest.json # PDF 来源文件名、页数、大小与 SHA256
│   ├── paper-analyses/         # 中文解析、教学示例、实验数据与定位链接
│   │   ├── foundations.js     # Transformer、BERT、LoRA
│   │   ├── training.js        # QLoRA、InstructGPT、DPO
│   │   └── agents.js          # ReAct、RAG
│   └── interviews/
│       ├── transformer.js
│       ├── rl.js
│       ├── sft.js
│       └── agent.js
├── scripts/
│   ├── sync-algorithms.mjs    # 校验、同步及调整站点命名
│   ├── algorithms-site.js    # 训练场导航、工具菜单与概况布局适配
│   ├── algorithms-runtime.js # 内嵌 Python 启动与错误处理适配
│   ├── check-algorithms-runtime.mjs # 浏览器运行时回归检查
│   ├── check-papers.mjs       # 论文阅读交互与布局检查
│   └── check-layout.mjs       # 七个普通页面的导航、菜单与响应式检查
└── NOTICE.md                 # 原始来源与许可说明
```

根目录只保留首页 `index.html`，子页面归入 `pages/`，其中四个面试专题统一放在 `pages/interviews/`。样式、交互和学习内容分别放在 `assets/css/`、`assets/js/` 和 `data/`。首页、算法页、论文页和面试页共用顶部导航，提供概览、算法、AI 专题和论文四个入口；面试页文件名不变，同一站点的浏览器题库继续沿用原存储键。

## 页面结构

各级页面分工如下，专题明细只出现在所属目录中：

```text
首页：三个学习入口
├── 算法训练场：题目目录 → 代码练习与评测
├── AI 专题：四个主题 → 章节与题目 → 展开答案
└── 论文阅读：合集与论文列表 → 阅读导引/解析 → 中文 PDF 或英文原文
```

首页、AI 目录和论文列表采用一个顶栏和一个页面标题，主题切换统一放在顶栏。AI 详情页用“返回 AI 专题”回到父级目录，页内导航只负责章节定位；窄屏时章节目录与题库管理并排，按需展开。四个专题集中在 `pages/interviews.html`，旧的 `index.html#interviews` 链接会转到新目录。

论文页的主题与阅读状态收进“筛选”，备份与阅读提示收进“阅读工具”；窄屏用带有可访问名称的省略号按钮呈现工具，与合集处于同一行。论文总量放在合集按钮中，列表结果数反映当前筛选，首次进入不自动展开正文。

导航、颜色、字体、间距和深浅模式共用站点样式。菜单采用原生 `details`，支持键盘展开、Escape 关闭、移出焦点或点击外部关闭；列表正文仍可通过原有链接直接定位。算法页沿用相同品牌和四项导航，并保留独立的吸顶工作区。

运行 `node scripts/check-layout.mjs` 检查首页、AI 专题目录、论文页和四个面试页在 320、390、768、1440 像素宽度下的深浅色布局、导航可见性、菜单展开和键盘操作，并保存截图；同时检查唯一标题与主题按钮、目录内容归属、父级返回、实际导航跳转和旧链接兼容。设置 `BASE_URL=https://yqstar.github.io` 可检查已发布页面。检查使用 Chrome 视口模拟，不代表真实手机或其他浏览器测试。

## 本地预览

在项目根目录运行：

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8000/>。不需要 npm 或构建步骤，也可以直接用浏览器打开 `index.html`。

## 维护面试题

编辑 `data/interviews/` 下对应主题的 JavaScript 文件，修改其中的 `const DATA` 对象。面试页通过普通 `<script>` 加载题库与 `assets/js/interview.js`，再调用 `initInterview(DATA)`，支持直接通过 `file://` 打开。

调整共用外观时编辑 `assets/css/site.css`，面试页专属样式编辑 `assets/css/interview.css`。页面内的导入、导出和本地存储用于浏览器中的学习数据，不会写回仓库文件；需要更新站点题库时仍应修改对应的 `data/interviews/*.js`。

## 论文阅读

`pages/papers.html` 默认展示 39 篇已完成的中文译文，共 543 页，覆盖推荐排序、序列建模、生成式推荐、模型架构和 Agent。每篇先展示研究问题、核心方法、精读线索，再提供“阅读中文 PDF”与“英文原文”链接。中文 PDF 从已有翻译成果原样复制，英文原文使用外链；页面不会预加载 PDF 文件。

通过合集按钮可切换到原有 8 篇经典解析，或浏览全部 47 篇。经典解析保留结论速览、流程图、方法拆解、核心公式、教学示例、实验对照、贡献、局限和原文定位；教学例与论文实验明确区分。每篇均可记录阅读进度和笔记。

可搜索论文标题、作者和解析全文，通过“筛选”按方向或阅读状态查找，并记录进度与个人笔记。“阅读工具”提供导入和导出入口。原有笔记 ID、存储键和 JSON 备份格式保持兼容。

阅读记录使用独立的 `study-hub:papers:v1` 存储键，可导出 JSON 备份，再通过导入恢复或迁移到其他浏览器。导入时先显示覆盖范围，经确认后合并；不在备份中的笔记会保留。导入上限为 8 MiB，可容纳 47 篇的完整中文笔记。记录不会自动同步到其他设备，也不会写回仓库。英文原文链接需要联网打开；通过 `file://` 使用时，本地存储行为取决于浏览器策略。

基础资料位于 `data/papers.js`，解析正文位于 `data/paper-analyses/`，按同一论文 ID 对应。解析须核对全文与实验表，注明测试/开发集、基线及数值口径，不能把教学示例写成论文结果。页面行为和外观分别位于 `assets/js/papers.js` 与 `assets/css/papers.css`。增加或替换论文时应核对标题、作者、年份和来源链接，并同步首页的论文数量。

译文元数据位于 `data/translated-papers.js`，中文 PDF 位于 `assets/papers/zh/`。来源清单 `data/paper-translations-manifest.json` 记录原始目录内的相对路径、文件大小、页数与 SHA256，便于核对复制结果；不包含英文 PDF、LaTeX 构建文件或本机工具链。替换译文时须同步清单与页面元数据。少数原文网页已更新标题或实验数字，导引以所收录的本地译稿对应版本为依据。

运行 `node scripts/check-papers.mjs` 可检查 39 份中文 PDF 与清单的一致性、英文来源链接、合集与关键词筛选、译文笔记、8 篇经典解析、目录跳转与焦点、备份导入导出、存储不可用时的行为、页面导航及桌面和窄屏的深浅色布局。若本机存在原始译文目录，还会逐一比对来源文件；可用 `ARXIV_PAPER_DIR` 指定目录。检查使用临时 Chrome 配置，不读取个人浏览器记录；`CHROME_PATH` 可指定浏览器安装位置。

## 深浅模式

首页、四个面试页、论文阅读页及算法训练场均可通过右上角的太阳 / 月亮按钮切换深浅模式。同一站点来源下共用主题偏好，刷新、跳转和已打开的标签页保持同步。首次使用跟随系统；已有算法页主题会迁移为全站偏好，手动选择后优先使用所选模式。本地存储不可用时，当前页面仍可切换。

共享逻辑位于 `assets/js/theme.js`，使用独立的 `study-hub:theme` 存储键。算法页的旧设置与备份格式继续兼容；恢复备份中的主题也会应用到全站。算法页内嵌这份脚本以保留独立离线使用，修改后需运行下面的同步命令重新生成。直接通过 `file://` 打开时，跨文件共享偏好取决于浏览器的本地文件存储策略。

## 算法训练场

`pages/algorithms.html` 基于未压缩单文件版本，约 20 MiB，直接保留 HTML、CSS 和 JavaScript，内嵌 100 道题、题解、评测器、格式化器和 Python 运行时。首次访问需下载完整页面，保存文件后可直接离线打开；使用支持原生 Base64 的现代浏览器。

站点统一使用「算法训练场」名称，顶部使用统一的 Study Hub 品牌和四项栏目导航，右侧保留工具和主题切换；手机上分为品牌工具行与栏目导航行。默认题面展开、代码模式和学习记录备份集中到“工具”，统计收进“学习概况”；继续练习、随机题和搜索筛选保持直接可用。题目与评测逻辑沿用源文件，Python 加载增加浏览器兼容适配；历史存储键和 JSON 备份格式保持兼容，已有代码、笔记和进度在同一站点来源下继续可用。

Python Worker 通过内嵌 `data:` 地址导入 JavaScript 模块，Wasm 和标准库直接从内存提供，不再依赖 Worker 内的 Blob URL 读取。Wasm 使用已缓存字节进行编译，并保留 Pyodide 的初始化钩子；底层编译失败会立即传回页面，避免被加载器吞掉后只显示“启动超时”。状态栏显示当前启动阶段，失败后可点击运行重试。运行代码和格式化无需联网。

题目示例保留原始内容，来源和许可集中在 [NOTICE.md](NOTICE.md)。

### 更新离线页面

当前固定上游提交 `5828bd249feede51f752ee42f9ee92254ac84933`，源文件的 SHA-256 为 `a745776bc832a1080ebfa1ad6f266fa39aaf5d2e01405e7ef55803570d7f9870`。

下载方式见 [来源说明](NOTICE.md)，也可直接使用本地原始文件：

```bash
node scripts/sync-algorithms.mjs /path/to/lc_offline.html
```

同步脚本校验来源文件，替换站点展示名称，加入 `scripts/algorithms-site.js` 中的导航和布局适配，并内嵌共享导航、主题脚本和 `scripts/algorithms-runtime.js` 中的启动适配，直接写入 `pages/algorithms.html`，不压缩或添加解压启动页。题库、编辑器、学习记录存储和 Python 评测逻辑保留源文件实现，运行时不依赖外部脚本。修改布局、共享导航、共享主题或运行时适配后也需重新运行同步命令。升级上游版本时先检查文案、导航、主题和运行时适配点，再一起更新提交号与校验值。

### 运行时回归检查

安装 Node.js 和 Chrome 后运行（无需 npm 依赖）：

```bash
node scripts/check-algorithms-runtime.mjs
```

脚本使用临时浏览器配置，检查 HTTP 和独立 `file://` 页面的 Python 启动、判题、格式化、启动失败重试及运行超时后恢复。测试禁止 Worker 请求外部 HTTP(S) 资源及读取 Blob URL，并模拟流式编译不可用、Wasm 编译失败，验证兼容加载和底层错误上报。默认使用 macOS Chrome 路径，其他安装位置可通过 `CHROME_PATH` 指定；这些兼容性模拟不代表实际 Safari 测试。
