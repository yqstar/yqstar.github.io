/* Shared reader for the four AI interview topics. Default data lives in data/interviews/. */
const INTERVIEW_TOPICS = [
  { key: 'transformer', name: 'Transformer' },
  { key: 'rl', name: 'RL' },
  { key: 'sft', name: 'SFT' },
  { key: 'agent', name: 'Agent' },
];
const IV = { defaultData: null, current: null, toolbar: null, fileInput: null, sections: [], frame: null, invalidStored: false };
const IV_DESKTOP = window.matchMedia('(min-width: 901px)');
const IV_CHEVRON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 10 4 4 4-4"/></svg>';

function esc(value) {
  return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}


function ivBreadcrumbSeparator() {
  return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>';
}
function ivTopicNavigation(topic) {
  return '<nav class="interview-topics" aria-label="AI 面试专题">' + INTERVIEW_TOPICS.map(item =>
    '<a href="' + item.key + '-interview.html"' + (item.key === topic.key ? ' aria-current="page"' : '') + '>' + esc(item.name) + '</a>'
  ).join('') + '</nav>';
}

function ivTopic() {
  return INTERVIEW_TOPICS.find(topic => topic.key === document.documentElement.dataset.topic) || INTERVIEW_TOPICS[0];
}

function ivSectionName(title) {
  return title.replace(/^[一二三四五六七八九十]+、\s*/, '');
}

function renderTags(tags = []) {
  const order = { hot: 0, core: 1 };
  return '<span class="q-tags">' + [...tags].sort((first, second) => (order[first.cls] ?? 2) - (order[second.cls] ?? 2)).map(tag => {
    const variant = ['hot', 'core'].includes(tag.cls) ? ' ' + tag.cls : '';
    return '<span class="q-tag' + variant + '">' + esc(tag.text) + '</span>';
  }).join('') + '</span>';
}

function renderQuestion(question) {
  return '<details class="q-card" id="' + esc(question.id) + '">' +
    '<summary class="q-header"><span class="q-num" aria-hidden="true">' + esc(String(question.num).padStart(2, '0')) + '</span>' +
    '<h3 class="q-text"><span class="q-title">' + esc(question.title) + '</span></h3>' +
    '<span class="q-arrow"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14"/><path class="q-plus-vertical" d="M12 5v14"/></svg></span>' + renderTags(question.tags) + '</summary>' +
    '<div class="q-body"><div class="answer-content">' + question.body + '</div></div></details>';
}

function renderSection(section, index) {
  return '<section class="section" id="' + esc(section.id) + '" aria-labelledby="iv-heading-' + esc(section.id) + '">' +
    '<div class="section-title"><span class="section-index" aria-hidden="true">' + String(index + 1).padStart(2, '0') + '</span>' +
    '<h2 id="iv-heading-' + esc(section.id) + '">' + esc(ivSectionName(section.title)) + '</h2>' +
    '<span class="s-count">' + section.questions.length + ' 题</span></div>' + section.questions.map(renderQuestion).join('') + '</section>';
}

function renderInterview(data, focusHash = true) {
  // The toolbar is reused across imports; never carry an open popup into a new view.
  if (IV.toolbar) IV.toolbar.open = false;
  const topic = ivTopic();
  const count = data.sections.reduce((total, section) => total + section.questions.length, 0);
  const toc = data.sections.map((section, index) => '<li><a href="#' + esc(section.id) + '">' +
    '<span class="toc-number" aria-hidden="true">' + String(index + 1).padStart(2, '0') + '</span>' +
    '<span>' + esc(ivSectionName(section.toc || section.title)) + '</span></a></li>').join('');

  document.body.classList.add('site-shell');
  document.getElementById('app').innerHTML =
    '<a class="skip-link" href="#iv-content">跳转到题目</a>' + window.StudyHubNavigation.markup('../../', 'interviews') +
    '<main class="page-width"><div class="workspace-bar"><nav class="breadcrumbs" aria-label="当前位置"><a href="../../index.html#interviews">AI 专题</a>' + ivBreadcrumbSeparator() +
    '<span aria-current="page">' + esc(topic.name) + '</span></nav><div class="workspace-actions"><button class="theme-toggle" data-theme-toggle type="button" hidden></button></div></div>' +
    '<header class="page-header"><h1>' + esc(data.title) + '</h1><p class="subtitle">' + esc(data.subtitle) + '</p>' +
    '<p class="page-meta">' + count + ' 道题<span aria-hidden="true"> · </span>' + data.sections.length + ' 个章节</p></header>' + ivTopicNavigation(topic) +
    '<p id="iv-message" class="iv-message" role="status" aria-live="polite" hidden></p>' +
    '<div class="reader-layout"><aside class="toc"><details class="toc-panel"' + (IV_DESKTOP.matches ? ' open' : '') + '><summary><span>章节目录</span><span class="toc-summary-note">' + IV_CHEVRON + '</span></summary>' +
    '<nav aria-label="章节目录"><ol class="toc-list">' + toc + '</ol></nav></details></aside><div id="iv-toolbar-slot"></div>' +
    '<div class="questions" id="iv-content" tabindex="-1">' + data.sections.map(renderSection).join('') + (data.sections.length ? '' : '<p class="empty-state">当前题库没有章节。</p>') + '</div></div>' +
    '<footer class="site-footer"><span>Study Hub<span class="footer-divider" aria-hidden="true">/</span>专注练习，持续积累。</span><span class="footer-credit">' + data.footer + '</span></footer></main>';

  window.StudyHubTheme?.refreshControls();

  // Keep home links in previously saved or imported footers working after the move.
  document.querySelectorAll('.footer-credit a[href]').forEach(link => {
    const href = link.getAttribute('href');
    if (/^(?:\.\/)?index\.html(?:[?#]|$)/.test(href)) {
      link.setAttribute('href', '../../' + href.replace(/^\.\//, ''));
    }
  });
  ivFormatAnswers();
  // Keep tables and long formulas inside their own scroll areas on small screens.
  document.querySelectorAll('.q-body table').forEach(table => {
    table.querySelectorAll('tr:first-child > th:not([scope])').forEach(cell => cell.setAttribute('scope', 'col'));
    const wrapper = document.createElement('div');
    wrapper.className = 'table-scroll';
    wrapper.tabIndex = 0;
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label', '对比表格，可横向滚动');
    table.before(wrapper);
    wrapper.append(table);
  });
  document.querySelectorAll('.q-body pre').forEach(pre => {
    pre.tabIndex = 0;
    pre.setAttribute('aria-label', '代码或公式，可横向滚动');
  });
  const panel = document.querySelector('.toc-panel');
  panel.addEventListener('toggle', () => {
    if (!IV_DESKTOP.matches && panel.open) {
      IV.toolbar.open = false;
      ivPositionToc();
    }
  });
  panel.addEventListener('focusout', event => {
    if (!IV_DESKTOP.matches && !panel.contains(event.relatedTarget)) panel.open = false;
  });
  document.querySelectorAll('.toc-list a').forEach(link => link.addEventListener('click', () => {
    if (!IV_DESKTOP.matches) panel.open = false;
    // Selecting the current fragment should still return to its heading.
    if (location.hash === link.getAttribute('href')) ivOpenHash();
  }));
  IV.sections = [...document.querySelectorAll('.section')];
  ivBuildToolbar();
  ivRefreshToolbar(IV.current !== IV.defaultData || IV.invalidStored);
  ivOpenHash({ focus: focusHash, scroll: focusHash });
  ivUpdateToc();
}

function ivFormatAnswers() {
  // Imported backups may still use the original div labels and bold paragraphs.
  // Normalize only the rendered DOM, preserving the stored and exported content.
  document.querySelectorAll('.answer-content').forEach(answer => {
    answer.querySelectorAll('[style]').forEach(element => element.removeAttribute('style'));
    answer.querySelectorAll('.answer-label:not(h4)').forEach(label => {
      const heading = document.createElement('h4');
      heading.className = 'answer-label';
      heading.append(...label.childNodes);
      label.replaceWith(heading);
    });
    answer.querySelectorAll('.answer-section > p').forEach(paragraph => {
      const children = [...paragraph.childNodes].filter(node => node.nodeType !== 3 || node.textContent.trim());
      if (children.length !== 1 || children[0].nodeType !== 1 || !['STRONG', 'B'].includes(children[0].tagName)) return;
      const heading = document.createElement('h5');
      heading.className = 'answer-subheading';
      heading.append(...children[0].childNodes);
      paragraph.replaceWith(heading);
    });
  });
}

function ivOpenHash(options = {}) {
  let id;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { return false; }
  const target = id && document.getElementById(id);
  if (!target || !target.matches('.q-card, .section')) return false;
  if (target.matches('.q-card')) target.open = true;
  requestAnimationFrame(() => {
    if (options.scroll !== false) target.scrollIntoView({ block: 'start' });
    const focusTarget = target.matches('.q-card') ? target.querySelector('summary') : target.querySelector('h2');
    if (focusTarget && options.focus !== false) {
      if (focusTarget.tagName !== 'SUMMARY') focusTarget.tabIndex = -1;
      focusTarget.focus({ preventScroll: true });
    }
    ivUpdateToc();
  });
  return true;
}

function ivPositionToc() {
  const panel = document.querySelector('.toc-panel');
  if (!panel) return;
  const navigation = panel.querySelector('nav');
  if (IV_DESKTOP.matches) { navigation.style.maxHeight = ''; return; }
  if (!panel.open) return;
  const trigger = panel.querySelector('summary').getBoundingClientRect();
  const below = innerHeight - trigger.bottom - 20;
  const above = trigger.top - 20;
  const placeAbove = below < navigation.scrollHeight && above > below;
  panel.dataset.placement = placeAbove ? 'above' : 'below';
  navigation.style.maxHeight = Math.max(0, placeAbove ? above : below) + 'px';
}

function ivUpdateToc() {
  IV.frame = null;
  let active = IV.sections[0];
  for (const section of IV.sections) {
    if (section.getBoundingClientRect().top <= 140) active = section;
  }
  document.querySelectorAll('.toc-list a').forEach(link => {
    if (active && link.getAttribute('href') === '#' + active.id) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
window.addEventListener('hashchange', ivOpenHash);
window.addEventListener('scroll', () => { if (IV.frame === null) IV.frame = requestAnimationFrame(ivUpdateToc); }, { passive: true });
IV_DESKTOP.addEventListener('change', () => {
  const panel = document.querySelector('.toc-panel');
  if (panel) panel.open = IV_DESKTOP.matches;
  ivPositionToc();
});

function ivKey() { return 'ivdata:' + (location.pathname.split('/').pop() || 'page'); }

function ivValid(data) {
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const scalar = value => typeof value === 'string' || typeof value === 'number' && Number.isFinite(value);
  if (!object(data) || !['title', 'highlight', 'subtitle', 'icon', 'footer'].every(key => typeof data[key] === 'string')) return false;
  if (!Array.isArray(data.sections) || data.sections.length > 100) return false;
  if (data.stats !== undefined && (!Array.isArray(data.stats) || !data.stats.every(stat => object(stat) && scalar(stat.num) && typeof stat.label === 'string'))) return false;
  const ids = new Set();
  const uniqueId = (id, pattern) => typeof id === 'string' && pattern.test(id) && !ids.has(id) && !!ids.add(id);
  return data.sections.every(section => object(section) && uniqueId(section.id, /^s\d+$/) && typeof section.title === 'string' &&
    (section.toc === undefined || section.toc === null || typeof section.toc === 'string') && Array.isArray(section.questions) && section.questions.length <= 500 &&
    section.questions.every(question => object(question) && uniqueId(question.id, /^q\d+$/) && scalar(question.num) && typeof question.title === 'string' && typeof question.body === 'string' &&
      (question.tags === undefined || Array.isArray(question.tags) && question.tags.every(tag => object(tag) && typeof tag.text === 'string' && (tag.cls === undefined || typeof tag.cls === 'string')))));
}

function ivSanitize(html) {
  const template = document.createElement('template');
  template.innerHTML = html;
  const allowed = new Set(['DIV', 'SPAN', 'P', 'BR', 'STRONG', 'EM', 'B', 'I', 'U', 'S', 'UL', 'OL', 'LI', 'PRE', 'CODE', 'TABLE', 'THEAD', 'TBODY', 'TFOOT', 'TR', 'TH', 'TD', 'A', 'BLOCKQUOTE', 'HR', 'H2', 'H3', 'H4', 'H5', 'H6', 'SUP', 'SUB']);
  template.content.querySelectorAll('script,style,iframe,object,embed,form,svg,math,template').forEach(element => element.remove());
  template.content.querySelectorAll('*').forEach(element => {
    if (!allowed.has(element.tagName)) { element.replaceWith(...element.childNodes); return; }
    for (const attribute of [...element.attributes]) {
      const name = attribute.name.toLowerCase();
      const safe = ['class', 'title', 'colspan', 'rowspan', 'scope', 'lang', 'dir'].includes(name) || element.tagName === 'A' && name === 'href';
      if (!safe) element.removeAttribute(attribute.name);
    }
    if (element.hasAttribute('href')) {
      const href = element.getAttribute('href');
      try {
        const url = new URL(href, location.href);
        const relativeFile = url.protocol === 'file:' && !/^[a-z][a-z\d+.-]*:|^\/\//i.test(href.trim());
        if (!['http:', 'https:', 'mailto:'].includes(url.protocol) && !relativeFile) element.removeAttribute('href');
      } catch { element.removeAttribute('href'); }
    }
  });
  return template.innerHTML;
}

function ivSanitizeData(data) {
  data.sections.forEach(section => section.questions.forEach(question => { question.body = ivSanitize(question.body); }));
  data.footer = ivSanitize(data.footer);
  return data;
}

function ivLoad() {
  IV.invalidStored = false;
  try {
    const raw = localStorage.getItem(ivKey());
    if (raw === null) return null;
    const data = JSON.parse(raw);
    if (!ivValid(data)) throw new Error('Invalid question data');
    return ivSanitizeData(data);
  } catch {
    IV.invalidStored = true;
    return null;
  }
}
function ivSave(data) {
  try { localStorage.setItem(ivKey(), JSON.stringify(data)); return true; } catch { return false; }
}
function ivClear() {
  try { localStorage.removeItem(ivKey()); return true; } catch { return false; }
}
function ivFileName() { return (location.pathname.split('/').pop() || 'interview').replace(/\.html?$/i, '') + '-data.json'; }
function ivDownload(filename, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function ivExport() { ivDownload(ivFileName(), JSON.stringify(IV.current, null, 2)); }
function ivMessage(message, error = false) {
  IV.toolbar.open = false;
  const output = document.getElementById('iv-message');
  output.textContent = message;
  output.classList.toggle('is-error', error);
  output.hidden = false;
}
function ivImport(file) {
  if (!file) return;
  if (file.size > 5 * 1024 * 1024) { ivMessage('导入失败：题库文件不能超过 5 MiB。', true); return; }
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      if (!ivValid(data)) throw new Error('题库结构或字段格式不正确，请使用本站导出的 JSON 格式。');
      ivSanitizeData(data);
      if (!ivSave(data)) throw new Error('无法保存到本地存储，现有题库未更改。');
      IV.current = data;
      IV.invalidStored = false;
      renderInterview(data, false);
      ivMessage('题库已导入并保存，刷新后仍可使用。');
      IV.toolbar.querySelector('summary').focus();
    } catch (error) { ivMessage('导入失败：' + (error.message || 'JSON 解析错误'), true); }
  };
  reader.onerror = () => ivMessage('读取文件失败，请重新选择文件。', true);
  reader.readAsText(file, 'utf-8');
}
function ivReset() {
  if (!confirm('重置为内置默认题库？当前导入的自定义题库将清除。')) return;
  if (!ivClear()) { ivMessage('无法清除本地存储，题库未重置。', true); return; }
  IV.current = IV.defaultData;
  IV.invalidStored = false;
  renderInterview(IV.current, false);
  ivMessage('已恢复内置题库。');
  IV.toolbar.querySelector('summary').focus();
}
function ivRefreshToolbar(custom) {
  IV.toolbar.querySelector('.iv-reset').disabled = !custom;
  IV.toolbar.querySelector('.iv-state').textContent = IV.invalidStored ? '内置题库 · 本地数据异常' : custom ? '自定义题库' : '内置题库';
}
function ivPositionToolbar() {
  if (!IV.toolbar?.open) return;
  const toolbar = IV.toolbar;
  const popup = toolbar.querySelector('.iv-tools-content');
  const trigger = toolbar.querySelector('summary').getBoundingClientRect();
  const below = window.innerHeight - trigger.bottom - 20;
  const above = trigger.top - 20;
  const placeAbove = below < popup.scrollHeight && above > below;
  toolbar.dataset.placement = placeAbove ? 'above' : 'below';
  popup.style.maxHeight = Math.max(0, placeAbove ? above : below) + 'px';
}
function ivBuildToolbar() {
  if (!IV.toolbar) {
    const toolbar = document.createElement('details');
    toolbar.className = 'iv-toolbar';
    toolbar.innerHTML = '<summary class="iv-menu-trigger">题库管理' + IV_CHEVRON + '</summary><div class="iv-tools-content"><span class="iv-state"></span><div class="iv-actions" role="group" aria-label="题库导入与导出">' +
      '<button type="button" class="iv-btn iv-export">导出题库</button><button type="button" class="iv-btn iv-import">导入题库</button>' +
      '<button type="button" class="iv-btn iv-reset">恢复默认题库</button></div></div>';
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,application/json';
    input.hidden = true;
    input.addEventListener('change', () => { ivImport(input.files?.[0]); input.value = ''; });
    toolbar.querySelector('.iv-export').addEventListener('click', () => { ivExport(); toolbar.open = false; toolbar.querySelector('summary').focus(); });
    toolbar.querySelector('.iv-import').addEventListener('click', () => { input.click(); toolbar.open = false; toolbar.querySelector('summary').focus(); });
    toolbar.querySelector('.iv-reset').addEventListener('click', ivReset);
    toolbar.addEventListener('toggle', () => {
      if (toolbar.open && !IV_DESKTOP.matches) document.querySelector('.toc-panel').open = false;
      ivPositionToolbar();
    });
    // This is a non-modal disclosure: Tab follows document order and exits freely.
    toolbar.addEventListener('focusout', event => {
      if (!toolbar.contains(event.relatedTarget)) toolbar.open = false;
    });
    toolbar.append(input);
    IV.toolbar = toolbar;
    IV.fileInput = input;
  }
  document.getElementById('iv-toolbar-slot').append(IV.toolbar);
}
document.addEventListener('click', event => {
  if (IV.toolbar?.open && !IV.toolbar.contains(event.target)) IV.toolbar.open = false;
  const panel = document.querySelector('.toc-panel');
  if (!IV_DESKTOP.matches && panel?.open && !panel.contains(event.target)) panel.open = false;
});
document.addEventListener('keydown', event => {
  const panel = document.querySelector('.toc-panel');
  if (event.key === 'Escape' && !IV_DESKTOP.matches && panel?.open) {
    event.preventDefault();
    panel.open = false;
    panel.querySelector('summary').focus();
  }
  if (event.key === 'Escape' && IV.toolbar?.open) {
    event.preventDefault();
    IV.toolbar.open = false;
    IV.toolbar.querySelector('summary').focus();
  }
});
window.addEventListener('resize', () => { ivPositionToolbar(); ivPositionToc(); });
window.addEventListener('scroll', () => {
  if (IV.toolbar?.open) ivPositionToolbar();
  ivPositionToc();
}, { passive: true });
function initInterview(defaultData) {
  IV.defaultData = defaultData;
  IV.current = ivLoad() || defaultData;
  renderInterview(IV.current);
  if (IV.invalidStored) ivMessage('本地题库无法读取，已显示内置内容。可导入有效备份或重置题库。', true);
}
