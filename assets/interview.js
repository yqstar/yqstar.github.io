/* ============================================================
   Algorithm & AI Study Hub — interview page renderer
   Each page defines `const DATA = {...}` then calls renderInterview(DATA).
   DATA schema:
     {
       icon, title, highlight, subtitle,
       stats: [{num, label}],
       sections: [{
         id, icon, title, count,
         questions: [{ id, num, title, tags:[{text,cls}], body }]
       }],
       footer  // raw HTML string
     }
   `body` is the raw inner HTML of the .q-body (answer-sections etc.),
   preserved verbatim from the original hand-authored markup.
   ============================================================ */

function toggleQ(header) {
  header.parentElement.classList.toggle('open');
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderTags(tags) {
  if (!tags || !tags.length) return '';
  return '<div class="q-tags">' + tags.map(function (t) {
    var cls = t.cls ? ' q-tag ' + t.cls : ' q-tag';
    return '<span class="' + cls.trim() + '">' + esc(t.text) + '</span>';
  }).join('') + '</div>';
}

function renderQuestion(q) {
  return '' +
    '<div class="q-card" id="' + q.id + '">' +
      '<div class="q-header" onclick="toggleQ(this)">' +
        '<span class="q-num">' + esc(q.num) + '</span>' +
        '<div class="q-text">' +
          '<h4>' + esc(q.title) + '</h4>' +
          renderTags(q.tags) +
        '</div>' +
        '<span class="q-arrow">▶</span>' +
      '</div>' +
      '<div class="q-body">' + q.body + '</div>' +
    '</div>';
}

function renderSection(s) {
  return '' +
    '<div class="section" id="' + s.id + '">' +
      '<div class="section-title">' +
        '<span class="s-icon">' + s.icon + '</span> ' + esc(s.title) +
        '<span class="s-count">' + esc(s.count) + '</span>' +
      '</div>' +
      (s.questions || []).map(renderQuestion).join('') +
    '</div>';
}

function renderInterview(data) {
  var stats = (data.stats || []).map(function (st) {
    return '<div class="stat-item"><div class="stat-num">' + esc(st.num) + '</div><div class="stat-label">' + esc(st.label) + '</div></div>';
  }).join('');

  var toc = (data.sections || []).map(function (s) {
    return '<li><a href="#' + s.id + '">' + esc(s.toc || s.title) + '</a></li>';
  }).join('');

  var sections = (data.sections || []).map(renderSection).join('');

  var html = '' +
    '<div class="page-header">' +
      '<span class="icon">' + data.icon + '</span>' +
      '<h1>' + esc(data.title) + ' <span class="gradient-text">' + esc(data.highlight) + '</span></h1>' +
      '<p class="subtitle">' + esc(data.subtitle) + '</p>' +
      '<a href="index.html" class="back-link">← 返回 Study Hub</a>' +
    '</div>' +
    '<div class="stats-bar">' + stats + '</div>' +
    '<div class="container">' +
      '<div class="toc">' +
        '<h3>📑 目录</h3>' +
        '<ul class="toc-list">' + toc + '</ul>' +
      '</div>' +
      sections +
    '</div>' +
    '<div class="footer">' + data.footer + '</div>';

  var app = document.getElementById('app');
  app.innerHTML = html;

  // Auto-open the question targeted by the URL hash (deep link).
  var hash = window.location.hash;
  if (hash) {
    var target = document.querySelector(hash);
    if (target) target.classList.add('open');
  }
}

// Also honour hash changes that happen after load.
window.addEventListener('hashchange', function () {
  var hash = window.location.hash;
  if (hash) {
    var target = document.querySelector(hash);
    if (target) target.classList.add('open');
  }
});

/* ============================================================
   Import / Export — per-page JSON, persisted to localStorage.
   Each page calls initInterview(DATA) instead of renderInterview(DATA).
   ============================================================ */
var IV = { defaultData: null, current: null, toolbar: null, fileInput: null };

function ivKey() {
  var p = location.pathname.split('/').pop() || 'page';
  return 'ivdata:' + p;
}

function ivValid(d) {
  return d && typeof d === 'object' && Array.isArray(d.sections);
}

function ivLoad() {
  try {
    var raw = localStorage.getItem(ivKey());
    if (raw) {
      var d = JSON.parse(raw);
      if (ivValid(d)) return d;
    }
  } catch (e) { /* ignore corrupt storage */ }
  return null;
}

function ivSave(data) {
  try { localStorage.setItem(ivKey(), JSON.stringify(data)); } catch (e) {}
}

function ivClear() {
  try { localStorage.removeItem(ivKey()); } catch (e) {}
}

function ivFileName() {
  var base = (location.pathname.split('/').pop() || 'interview').replace(/\.html?$/i, '');
  return base + '-data.json';
}

function ivDownload(filename, text) {
  var blob = new Blob([text], { type: 'application/json;charset=utf-8' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
}

function ivExport() {
  ivDownload(ivFileName(), JSON.stringify(IV.current, null, 2));
}

function ivImport(file) {
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function () {
    try {
      var data = JSON.parse(reader.result);
      if (!ivValid(data)) throw new Error('数据缺少 sections 字段');
      IV.current = data;
      ivSave(data);
      renderInterview(data);
      ivRefreshToolbar(true);
      alert('导入成功（已保存到本地，刷新仍生效）');
    } catch (e) {
      alert('导入失败：' + (e.message || 'JSON 解析错误'));
    }
  };
  reader.onerror = function () { alert('读取文件失败'); };
  reader.readAsText(file, 'utf-8');
}

function ivReset() {
  if (!confirm('重置为内置默认数据？当前导入/修改将清除。')) return;
  ivClear();
  IV.current = IV.defaultData;
  renderInterview(IV.defaultData);
  ivRefreshToolbar(false);
}

function ivRefreshToolbar(custom) {
  if (!IV.toolbar) return;
  var resetBtn = IV.toolbar.querySelector('.iv-reset');
  if (resetBtn) resetBtn.disabled = !custom;
  var badge = IV.toolbar.querySelector('.iv-state');
  if (badge) badge.textContent = custom ? '已加载自定义数据' : '默认数据';
}

function ivBuildToolbar(custom) {
  if (IV.toolbar) return;
  var bar = document.createElement('div');
  bar.className = 'iv-toolbar';
  bar.innerHTML =
    '<span class="iv-state">' + (custom ? '已加载自定义数据' : '默认数据') + '</span>' +
    '<button type="button" class="iv-btn iv-export">导出 JSON</button>' +
    '<button type="button" class="iv-btn iv-import">导入 JSON</button>' +
    '<button type="button" class="iv-btn iv-reset" ' + (custom ? '' : 'disabled') + '>重置</button>';

  bar.querySelector('.iv-export').addEventListener('click', ivExport);
  bar.querySelector('.iv-import').addEventListener('click', function () { IV.fileInput.click(); });
  bar.querySelector('.iv-reset').addEventListener('click', ivReset);

  var input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json,application/json';
  input.style.display = 'none';
  input.addEventListener('change', function () { ivImport(input.files && input.files[0]); input.value = ''; });

  document.body.appendChild(bar);
  document.body.appendChild(input);
  IV.toolbar = bar;
  IV.fileInput = input;
}

function initInterview(defaultData) {
  IV.defaultData = defaultData;
  var stored = ivLoad();
  IV.current = stored || defaultData;
  renderInterview(IV.current);
  ivBuildToolbar(!!stored);
}
