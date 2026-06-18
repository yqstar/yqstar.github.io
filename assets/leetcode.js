function parseSignature(code) {
  const m = code.match(/def\s+(\w+)\(self(?:,\s*([^)]*))?\)\s*(?:->\s*([^:]+))?:/);
  if (!m) return null;
  const methodName = m[1];
  const rawParams = m[2] ? m[2].trim() : '';
  const returnType = m[3] ? m[3].trim() : '';
  const params = [];
  if (rawParams) {
    let depth = 0, cur = '';
    for (const ch of rawParams) {
      if (ch === '[') depth++;
      if (ch === ']') depth--;
      if (ch === ',' && depth === 0) { params.push(cur.trim()); cur = ''; }
      else cur += ch;
    }
    if (cur.trim()) params.push(cur.trim());
  }
  const parsed = params.map(p => {
    const [name, type] = p.includes(':') ? p.split(':').map(s=>s.trim()) : [p, ''];
    return { name, type };
  });
  return { methodName, params: parsed, returnType };
}

function isListNode(type) { return /ListNode|Optional\[ListNode\]/.test(type); }
function isTreeNode(type) { return /TreeNode|Optional\[TreeNode\]/.test(type); }
function isVoid(rt) { return rt === 'None' || rt === ''; }

function genRunCode(sig, caseValues) {
  if (!sig || !caseValues.length) return '';
  const lines = ['sol = Solution()'];
  const argNames = [];
  sig.params.forEach((p, i) => {
    const val = (caseValues[i] || '').replace(/\bnull\b/g, 'None');
    if (isListNode(p.type)) {
      lines.push(`${p.name} = build_list(${val})`);
    } else if (isTreeNode(p.type)) {
      lines.push(`${p.name} = build_tree(${val})`);
    } else {
      lines.push(`${p.name} = ${val}`);
    }
    argNames.push(p.name);
  });

  const call = `sol.${sig.methodName}(${argNames.join(', ')})`;

  if (isVoid(sig.returnType)) {
    lines.push(call);
    const first = sig.params[0];
    if (first && isListNode(first.type)) lines.push(`print(to_list(${first.name}))`);
    else lines.push(`print(${first ? first.name : '"done"'})`);
  } else {
    lines.push(`result = ${call}`);
    if (isListNode(sig.returnType)) lines.push('print(to_list(result))');
    else if (isTreeNode(sig.returnType)) lines.push('print(tree_to_list(result))');
    else lines.push('print(result)');
  }
  return lines.join('\n');
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ================================================================
// STATE
// ================================================================
const CATS = [...new Set(DATA.map(p => p.cat))];
const CAT_COUNTS = {};
CATS.forEach(cat => { CAT_COUNTS[cat] = DATA.filter(p => p.cat === cat).length; });

let cur = null;
let solved = new Set(JSON.parse(localStorage.getItem('lc_done') || '[]'));
let pyodide = null;
let pyReady = false;
let pyLoading = false;
let activeCaseIdx = 0;
let curCases = [];
let curSig = null;

let editorCleanups = [];

// ================================================================
// LIST PAGE
// ================================================================
let activeCat = 'all';
let activeDiff = 'all';
let searchTimer = null;

function buildListPage() {
  buildCatSidebar();
  buildProblemList();
  updProgress();

  const input = document.getElementById('search-input');
  if (input) {
    input.addEventListener('input', function() {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(buildProblemList, 150);
    });
  }
}

function buildCatSidebar() {
  const nav = document.getElementById('cat-sidebar');
  let h = '<div class="cat-label">分类</div>';
  h += `<a class="${activeCat==='all'?'active':''}" onclick="setCatFilter('all')">全部<span class="cat-cnt">${DATA.length}</span></a>`;
  for (const cat of CATS) {
    h += `<a class="${activeCat===cat?'active':''}" onclick="setCatFilter('${cat}')">${cat}<span class="cat-cnt">${CAT_COUNTS[cat]}</span></a>`;
  }
  nav.innerHTML = h;
}

function buildProblemList() {
  const q = (document.getElementById('search-input')?.value || '').trim().toLowerCase();
  let list = DATA;
  if (activeCat !== 'all') list = list.filter(p => p.cat === activeCat);
  if (activeDiff !== 'all') list = list.filter(p => p.diff === activeDiff);
  if (q) list = list.filter(p => p.title.toLowerCase().includes(q) || String(p.id).includes(q));

  const container = document.getElementById('problem-list');
  if (!list.length) { container.innerHTML = '<div style="padding:32px;text-align:center;color:var(--text3);font-size:0.9em">无匹配题目</div>'; return; }
  let h = '';
  for (const p of list) {
    const dc = p.diff==='Easy'?'easy':p.diff==='Medium'?'medium':'hard';
    const dl = p.diff==='Easy'?'简单':p.diff==='Medium'?'中等':'困难';
    const done = solved.has(p.id);
    h += `<div class="problem-row" onclick="select(${p.id})">`;
    h += `<span class="p-check${done?' done':''}" onclick="event.stopPropagation();toggleDone(${p.id})">${done?'✓':''}</span>`;
    h += `<span class="p-num">${p.id}</span>`;
    h += `<span class="p-title">${p.title}</span>`;
    h += `<span class="p-cat">${p.cat}</span>`;
    h += `<span class="p-diff ${dc}">${dl}</span>`;
    h += `<span class="p-arrow">›</span>`;
    h += `</div>`;
  }
  container.innerHTML = h;
}

function setCatFilter(cat) {
  activeCat = cat;
  buildCatSidebar();
  buildProblemList();
}

function setDiffFilter(btn, diff) {
  activeDiff = diff;
  document.querySelectorAll('.diff-filter').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  buildProblemList();
}

function filterList() { buildProblemList(); }

function updProgress() {
  const d = solved.size, t = DATA.length;
  const fill = document.getElementById('progress-fill');
  const text = document.getElementById('progress-text');
  if (fill) fill.style.width = (d/t*100)+'%';
  if (text) text.textContent = `已完成 ${d} / ${t} 题`;
}

// ================================================================
// PROBLEM SELECTION & RENDERING
// ================================================================
function select(id) {
  cur = DATA.find(p => p.id === id); if (!cur) return;
  document.getElementById('list-page').style.display = 'none';
  const main = document.getElementById('main-content');
  main.style.display = 'flex';
  main.classList.add('split-mode');
  const pv = document.getElementById('problem-view');
  pv.style.display = '';
  renderProblem();
  const lp = document.getElementById('left-panel');
  if (lp) lp.scrollTop = 0;
  if (!pyReady && !pyLoading) initPy();
}

function genSkeleton(code) {
  const lines = code.split('\n');
  const result = [];
  for (const line of lines) {
    const trimmed = line.trimStart();
    if (trimmed.startsWith('class ') || trimmed.startsWith('def ')) {
      result.push(line);
      const indent = line.match(/^(\s*)/)[1];
      result.push(indent + '    pass');
    }
  }
  return result.join('\n');
}

function fmtDesc(raw) {
  let s = esc(raw);
  s = s.replace(/示例[：:]/g, '<div class="desc-section">示例：</div>');
  s = s.replace(/(输入[：:].*?)(?=\n|$)/g, '<div class="desc-example">$1</div>');
  s = s.replace(/\n/g, '<br>');
  return s;
}

function cleanupEditorListeners() {
  editorCleanups.forEach(fn => fn());
  editorCleanups = [];
}

function renderProblem() {
  cleanupEditorListeners();

  const p = cur;
  const v = document.getElementById('problem-view');
  const mainEl = document.getElementById('main-content');
  mainEl.classList.add('split-mode');
  v.classList.add('split-view');
  const dc = p.diff==='Easy'?'easy':p.diff==='Medium'?'medium':'hard';
  v.innerHTML = `
    <div class="left-panel" id="left-panel">
      <div class="left-nav">
        <button class="back-btn" onclick="goBack()">‹</button>
        <span class="nav-tab active" onclick="switchTab(event,'desc')">题目描述</span>
        <span class="nav-tab" onclick="switchTab(event,'idea')">解法</span>
      </div>
      <div class="left-body">
        <div id="tb-desc" class="tab-content active">
          <div class="problem-title">
            <span class="diff-dot ${dc}"></span>
            <h1>${p.id}. ${p.title}</h1>
          </div>
          <div class="problem-tags">
            <span class="label">标签：</span>
            <span class="tag-pill">${p.cat}</span>
          </div>
          <div class="problem-meta">
            <a href="${p.lc}" target="_blank">LeetCode ↗</a>
            <button class="done-btn ${solved.has(p.id)?'isdone':'undone'}" id="done-btn" onclick="toggleDone(${p.id})">${solved.has(p.id)?'✓ 已完成':'○ 标记完成'}</button>
          </div>
          <div class="desc-body">${fmtDesc(p.desc)}</div>
        </div>
        <div id="tb-idea" class="tab-content">
          <div class="problem-title">
            <span class="diff-dot ${dc}"></span>
            <h1>${p.id}. ${p.title}</h1>
          </div>
          <div class="desc-section" style="margin-top:0">解题思路</div>
          <div class="desc-body">${esc(p.idea).replace(/\n/g,'<br>')}</div>
          <div class="desc-section">参考代码</div>
          <div class="solution-code-wrap">
            <button class="solution-copy-btn" onclick="copySolution()">复制</button>
            <pre class="solution-code" id="solution-code">${esc(p.code)}</pre>
          </div>
        </div>
      </div>
    </div>
    <div class="resizer" id="resizer"></div>
    <div class="right-panel" id="right-panel">
      <div class="editor-toolbar">
        <div class="toolbar-left">
          <span class="toolbar-select">Python3</span>
          <span class="toolbar-select">核心代码</span>
          <span class="py-status"><span class="dot" id="py-dot"></span><span id="py-label">Loading...</span></span>
        </div>
        <div class="toolbar-right">
          <button class="toolbar-btn btn-run-sm" onclick="runAllCases()" id="btn-run-all" title="运行全部用例">全部运行</button>
          <button class="toolbar-btn btn-run-sm" onclick="run()" id="btn-run" title="Ctrl+Enter"><span class="spinner" id="run-spinner"></span>运行</button>
          <button class="toolbar-btn btn-reset" onclick="resetCode()">重置</button>
        </div>
      </div>
      <div class="editor-wrap">
        <div class="editor-body">
          <div class="editor-lines" id="line-nums"></div>
          <textarea id="code-editor" spellcheck="false"></textarea>
        </div>
      </div>
      <div class="test-panel">
        <div class="test-panel-header" onclick="toggleTestPanel()">
          <span>测试用例</span>
          <span class="chevron open" id="test-chevron">▾</span>
        </div>
        <div class="test-panel-body open" id="test-body">
          <div id="structured-test"></div>
          <textarea id="test-editor" spellcheck="false" style="display:none"></textarea>
        </div>
      </div>
      <div class="output-section">
        <div class="output-header" onclick="toggleOutputPanel()">
          <span>输出</span>
          <span class="chevron open" id="output-chevron">▾</span>
        </div>
        <div class="output-body open" id="output-body">
          <pre id="output-panel"></pre>
        </div>
      </div>
    </div>
  `;
  const ce = document.getElementById('code-editor');
  ce.value = genSkeleton(p.code);
  updateLineNums();

  function onScroll() { syncScroll(); }
  function onInput() { updateLineNums(); }
  function onKeydown(e) { handleTab(e); }

  ce.addEventListener('scroll', onScroll);
  ce.addEventListener('input', onInput);
  ce.addEventListener('keydown', onKeydown);
  editorCleanups.push(
    function() { ce.removeEventListener('scroll', onScroll); },
    function() { ce.removeEventListener('input', onInput); },
    function() { ce.removeEventListener('keydown', onKeydown); }
  );

  document.getElementById('output-panel').textContent = '';
  updatePyStatus();
  initResizer();

  curSig = parseSignature(p.code);
  const useTextarea = TEXTAREA_PROBLEMS.has(p.id) || !curSig || !curSig.params.length;
  if (useTextarea) {
    document.getElementById('structured-test').style.display = 'none';
    const te = document.getElementById('test-editor');
    te.style.display = ''; te.value = p.test || '';
    document.getElementById('btn-run-all').style.display = 'none';
  } else {
    document.getElementById('test-editor').style.display = 'none';
    document.getElementById('structured-test').style.display = '';
    const raw = TEST_CASES[p.id];
    curCases = raw ? raw.map(c => [...c]) : [curSig.params.map(() => '')];
    activeCaseIdx = 0;
    renderCaseUI();
  }
}

function goBack() {
  cleanupEditorListeners();
  document.getElementById('problem-view').style.display = 'none';
  document.getElementById('problem-view').classList.remove('split-view');
  const main = document.getElementById('main-content');
  main.classList.remove('split-mode');
  main.style.display = 'none';
  document.getElementById('list-page').style.display = '';
  cur = null;
  buildProblemList();
  updProgress();
}

function toggleTestPanel() {
  const b = document.getElementById('test-body');
  const c = document.getElementById('test-chevron');
  b.classList.toggle('open');
  c.classList.toggle('open');
}

function toggleOutputPanel() {
  const b = document.getElementById('output-body');
  const c = document.getElementById('output-chevron');
  b.classList.toggle('open');
  c.classList.toggle('open');
}

// ================================================================
// STRUCTURED TEST CASE UI
// ================================================================
function renderCaseUI() {
  if (!curSig) return;
  const container = document.getElementById('structured-test');
  let h = '<div class="case-tabs">';
  curCases.forEach((_, i) => {
    h += `<span class="case-tab${i===activeCaseIdx?' active':''}" onclick="selectCase(${i})">Case ${i+1}<span class="case-x" onclick="event.stopPropagation();removeCase(${i})">×</span></span>`;
  });
  h += `<span class="case-add" onclick="addCase()">+</span></div>`;
  h += '<div class="case-fields">';
  const vals = curCases[activeCaseIdx] || [];
  curSig.params.forEach((p, i) => {
    let typeHint = '';
    if (isListNode(p.type)) typeHint = ' (数组表示链表)';
    else if (isTreeNode(p.type)) typeHint = ' (数组表示树，null为空)';
    h += `<div><div class="case-field-label">${p.name}${typeHint} =</div>`;
    h += `<input class="case-field-input" data-idx="${i}" value="${esc(vals[i]||'')}" oninput="updateCaseValue(${i},this.value)" /></div>`;
  });
  h += '</div>';
  container.innerHTML = h;
}

function selectCase(idx) {
  saveCaseInputs();
  activeCaseIdx = idx;
  renderCaseUI();
}

function addCase() {
  saveCaseInputs();
  curCases.push(curSig.params.map(() => ''));
  activeCaseIdx = curCases.length - 1;
  renderCaseUI();
}

function removeCase(idx) {
  if (curCases.length <= 1) return;
  curCases.splice(idx, 1);
  if (activeCaseIdx >= curCases.length) activeCaseIdx = curCases.length - 1;
  renderCaseUI();
}

function updateCaseValue(paramIdx, val) {
  if (curCases[activeCaseIdx]) curCases[activeCaseIdx][paramIdx] = val;
}

function saveCaseInputs() {
  document.querySelectorAll('.case-field-input').forEach(input => {
    const idx = parseInt(input.dataset.idx);
    if (curCases[activeCaseIdx]) curCases[activeCaseIdx][idx] = input.value;
  });
}

function updateLineNums() {
  const ce = document.getElementById('code-editor');
  if (!ce) return;
  const lines = ce.value.split('\n').length;
  const ln = document.getElementById('line-nums');
  let h = '';
  for (let i=1; i<=lines; i++) h += `<span>${i}</span>`;
  ln.innerHTML = h;
}

function syncScroll() {
  const ce = document.getElementById('code-editor');
  const ln = document.getElementById('line-nums');
  if (ln) ln.scrollTop = ce.scrollTop;
}

function handleTab(e) {
  if (e.key === 'Tab') {
    e.preventDefault();
    const ta = e.target;
    const start = ta.selectionStart;
    ta.value = ta.value.substring(0, start) + '    ' + ta.value.substring(ta.selectionEnd);
    ta.selectionStart = ta.selectionEnd = start + 4;
    updateLineNums();
  }
}

function switchTab(evt, name) {
  const panel = evt.target.closest('.left-panel') || document;
  panel.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
  panel.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  evt.target.classList.add('active');
  const el = document.getElementById('tb-'+name);
  if (el) el.classList.add('active');
}

function toggleDone(id) {
  if (solved.has(id)) solved.delete(id);
  else solved.add(id);
  try { localStorage.setItem('lc_done', JSON.stringify([...solved])); } catch(e) {}
  updProgress();
  if (cur && cur.id === id) {
    const btn = document.getElementById('done-btn');
    if (btn) {
      const done = solved.has(id);
      btn.className = 'done-btn ' + (done ? 'isdone' : 'undone');
      btn.textContent = done ? '✓ 已完成' : '○ 标记完成';
    }
  } else {
    buildProblemList();
  }
}

function copyCode() {
  if (!cur) return;
  navigator.clipboard.writeText(document.getElementById('code-editor').value);
}

function resetCode() {
  if (!cur) return;
  document.getElementById('code-editor').value = genSkeleton(cur.code);
  updateLineNums();
  document.getElementById('output-panel').textContent = '';
  if (curSig && TEST_CASES[cur.id]) {
    curCases = TEST_CASES[cur.id].map(c => [...c]);
    activeCaseIdx = 0;
    renderCaseUI();
  }
}

function copySolution() {
  if (!cur) return;
  navigator.clipboard.writeText(cur.code).then(() => {
    const btn = document.querySelector('.solution-copy-btn');
    if (btn) { btn.textContent = '已复制'; setTimeout(() => btn.textContent = '复制', 1500); }
  });
}

// ================================================================
// PYODIDE — lazy loaded
// ================================================================
function loadPyodideScript() {
  return new Promise(function(resolve, reject) {
    if (typeof loadPyodide === 'function') { resolve(); return; }
    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js';
    s.onload = resolve;
    s.onerror = function() { reject(new Error('Failed to load Pyodide')); };
    document.head.appendChild(s);
  });
}

async function initPy() {
  if (pyLoading || pyReady) return;
  pyLoading = true;
  updatePyStatus();
  try {
    await loadPyodideScript();
    pyodide = await loadPyodide();
    await pyodide.runPythonAsync(`
import sys, io, traceback
from typing import List, Optional, Dict, Set, Tuple
from collections import defaultdict, Counter, deque, OrderedDict
import heapq, math, functools, itertools, random

sys.setrecursionlimit(500)

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val; self.next = next
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right
class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val; self.next = next; self.random = random

def build_list(arr):
    if not arr: return None
    head = ListNode(arr[0])
    cur = head
    for v in arr[1:]:
        cur.next = ListNode(v); cur = cur.next
    return head

def to_list(head):
    res = []
    while head: res.append(head.val); head = head.next
    return res

def build_tree(arr):
    if not arr: return None
    root = TreeNode(arr[0])
    q = deque([root])
    i = 1
    while q and i < len(arr):
        node = q.popleft()
        if i < len(arr) and arr[i] is not None:
            node.left = TreeNode(arr[i]); q.append(node.left)
        i += 1
        if i < len(arr) and arr[i] is not None:
            node.right = TreeNode(arr[i]); q.append(node.right)
        i += 1
    return root

def tree_to_list(root):
    if not root: return []
    res, q = [], deque([root])
    while q:
        node = q.popleft()
        if node:
            res.append(node.val)
            q.append(node.left); q.append(node.right)
        else:
            res.append(None)
    while res and res[-1] is None: res.pop()
    return res
`);
    pyReady = true;
  } catch(e) {
    console.error('Pyodide init failed:', e);
  }
  pyLoading = false;
  updatePyStatus();
}

function updatePyStatus() {
  const dot = document.getElementById('py-dot');
  const label = document.getElementById('py-label');
  if (!dot || !label) return;
  if (pyReady) { dot.className = 'dot ready'; label.textContent = 'Python ready'; }
  else if (pyLoading) { dot.className = 'dot loading'; label.textContent = 'Loading Pyodide...'; }
  else { dot.className = 'dot'; label.textContent = 'Python not loaded'; }
}

async function executePython(fullCode) {
  pyodide.globals.set('_user_code', fullCode);
  const result = await pyodide.runPythonAsync(`
import sys, io, traceback

_op_count = 0
_op_limit = 2000000

def _trace(frame, event, arg):
    global _op_count
    _op_count += 1
    if _op_count > _op_limit:
        raise RuntimeError("执行超时：代码运行时间过长（可能存在死循环）")
    return _trace

_out = io.StringIO()
sys.stdout = _out
try:
    sys.settrace(_trace)
    exec(_user_code)
    _res = _out.getvalue()
except Exception as e:
    _res = "Error: " + type(e).__name__ + ": " + str(e) + "\\n" + traceback.format_exc()
finally:
    sys.settrace(None)
    sys.stdout = sys.__stdout__
_res
`);
  return result;
}

async function run() {
  if (!pyReady) {
    if (!pyLoading) initPy();
    document.getElementById('output-panel').textContent = 'Pyodide 正在加载，请稍候...';
    document.getElementById('output-panel').className = '';
    return;
  }
  const btn = document.getElementById('btn-run');
  btn.classList.add('running');
  const code = document.getElementById('code-editor').value;

  let test = '';
  const te = document.getElementById('test-editor');
  if (te && te.style.display !== 'none') {
    test = te.value;
  } else if (curSig && curCases.length) {
    saveCaseInputs();
    test = genRunCode(curSig, curCases[activeCaseIdx] || []);
  }
  const full = code + '\n\n' + test;

  try {
    const result = await executePython(full);
    const panel = document.getElementById('output-panel');
    panel.textContent = result || '(no output)';
    panel.className = result && result.startsWith('Error:') ? 'error' : '';
  } catch(e) {
    document.getElementById('output-panel').textContent = 'JS Error: ' + e.message;
    document.getElementById('output-panel').className = 'error';
  }
  btn.classList.remove('running');
  const ob = document.getElementById('output-body');
  const oc = document.getElementById('output-chevron');
  if (ob && !ob.classList.contains('open')) { ob.classList.add('open'); oc.classList.add('open'); }
}

async function runAllCases() {
  if (!pyReady) {
    if (!pyLoading) initPy();
    document.getElementById('output-panel').textContent = 'Pyodide 正在加载，请稍候...';
    document.getElementById('output-panel').className = '';
    return;
  }
  if (!curSig || !curCases.length) { run(); return; }

  const btn = document.getElementById('btn-run-all');
  const btnRun = document.getElementById('btn-run');
  btn.classList.add('running');
  btnRun.classList.add('running');
  saveCaseInputs();

  const code = document.getElementById('code-editor').value;
  const outputs = [];

  for (let i = 0; i < curCases.length; i++) {
    const test = genRunCode(curSig, curCases[i] || []);
    const full = code + '\n\n' + test;
    try {
      const result = await executePython(full);
      outputs.push(`Case ${i+1}: ${(result || '(no output)').trim()}`);
    } catch(e) {
      outputs.push(`Case ${i+1}: Error - ${e.message}`);
    }
  }

  const panel = document.getElementById('output-panel');
  const text = outputs.join('\n');
  panel.textContent = text;
  panel.className = text.includes('Error') ? 'error' : '';

  btn.classList.remove('running');
  btnRun.classList.remove('running');
  const ob = document.getElementById('output-body');
  const oc = document.getElementById('output-chevron');
  if (ob && !ob.classList.contains('open')) { ob.classList.add('open'); oc.classList.add('open'); }
}

// ================================================================
// RESIZER
// ================================================================
function initResizer() {
  const resizer = document.getElementById('resizer');
  const left = document.getElementById('left-panel');
  const right = document.getElementById('right-panel');
  if (!resizer || !left || !right) return;

  let startX, startLeftW, startRightW;

  function onPointerMove(e) {
    const dx = e.clientX - startX;
    const totalW = startLeftW + startRightW;
    let newLeftW = startLeftW + dx;
    let newRightW = startRightW - dx;
    if (newLeftW < 280) { newLeftW = 280; newRightW = totalW - 280; }
    if (newRightW < 320) { newRightW = 320; newLeftW = totalW - 320; }
    left.style.flex = 'none';
    right.style.flex = 'none';
    left.style.width = newLeftW + 'px';
    right.style.width = newRightW + 'px';
  }

  function onPointerUp() {
    resizer.classList.remove('active');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    document.removeEventListener('pointermove', onPointerMove);
    document.removeEventListener('pointerup', onPointerUp);
  }

  resizer.addEventListener('pointerdown', function(e) {
    e.preventDefault();
    startX = e.clientX;
    startLeftW = left.offsetWidth;
    startRightW = right.offsetWidth;
    resizer.classList.add('active');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerup', onPointerUp);
  });
}

// ================================================================
// KEYBOARD SHORTCUTS
// ================================================================
document.addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    if (cur && pyReady) run();
  }
});

// ================================================================
// INIT
// ================================================================
buildListPage();
