/* Local paper notebook. Plain scripts also work when opened with file://. */
(() => {
  'use strict';
  const papers = window.STUDY_PAPERS;
  const storageKey = 'study-hub:papers:v1';
  const categories = { architecture: '模型架构', tuning: '高效微调', alignment: '偏好对齐', agents: 'Agent 与检索' };
  const statuses = { unread: '未开始', reading: '在读', done: '已读' };
  const paperIds = new Set(papers.map(paper => paper.id));
  const byId = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
  const arrow = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M7 7h10v10"/></svg>';
  const noteTemplate = '## 研究问题\n作者想解决什么？已有方法的不足是什么？\n\n## 核心方法\n用自己的话说明关键步骤。\n\n## 实验证据\n记录数据、基线、评价指标和关键消融。\n\n## 局限与疑问\n哪些结论还需要验证？\n\n## 我的理解\n用一句话总结，并联系已有知识。\n';
  let records = Object.create(null);
  let storageAvailable = true;
  let category = 'all';
  let statusFilter = 'all';
  let pendingImport = null;
  let importSequence = 0;

  function notice(message, error = false) {
    const element = byId('paper-message');
    element.textContent = message;
    element.className = 'paper-notice' + (error ? ' error' : '');
    element.hidden = false;
  }

  function disableStorage(message) {
    storageAvailable = false;
    byId('storage-warning').textContent = message + ' 当前修改仅保留在本次页面中，请导出备份。';
    byId('storage-warning').hidden = false;
  }

  function validateBackup(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value) || value.version !== 1
      || !value.records || typeof value.records !== 'object' || Array.isArray(value.records)) {
      throw new Error('请选择本专题导出的版本 1 JSON 备份。');
    }
    const next = Object.create(null);
    for (const [id, record] of Object.entries(value.records)) {
      if (!paperIds.has(id)) throw new Error('备份中包含不支持的论文记录。');
      if (!record || typeof record !== 'object' || Array.isArray(record)
        || typeof record.status !== 'string' || !Object.hasOwn(statuses, record.status)
        || typeof record.note !== 'string' || record.note.length > 20000) {
        throw new Error('论文记录格式不正确，或单篇笔记超过 20,000 字符。');
      }
      next[id] = { status: record.status, note: record.note };
    }
    return next;
  }

  function readStored() {
    const stored = localStorage.getItem(storageKey);
    return stored === null ? Object.create(null) : validateBackup(JSON.parse(stored));
  }

  try { records = readStored(); }
  catch { disableStorage('无法读取本机记录，已有存储不会被覆盖。'); }

  function recordFor(id) {
    return records[id] || { status: 'unread', note: '' };
  }

  function commit(patch) {
    // Merge only the edited records so a second tab's other notes are preserved.
    let current = records;
    if (storageAvailable) {
      try { current = readStored(); }
      catch { disableStorage('无法读取本机记录，已停止自动写入。'); }
    }
    records = Object.assign(Object.create(null), current, patch);
    if (storageAvailable) {
      try { localStorage.setItem(storageKey, JSON.stringify({ version: 1, records })); }
      catch { disableStorage('保存失败，浏览器存储可能不可用或已满。'); }
    }
  }

  function renderPaper(paper, index) {
    const record = recordFor(paper.id);
    return `<details class="paper-card" id="paper-${paper.id}" data-paper-id="${paper.id}">
      <summary class="paper-summary"><span class="paper-index" aria-hidden="true">${String(index + 1).padStart(2, '0')}</span>
        <div class="paper-summary-copy"><span class="paper-meta"><span class="paper-category">${categories[paper.category]}</span><span class="paper-meta-dot" aria-hidden="true"></span><span>${paper.year}</span></span>
          <h3>${esc(paper.shortTitle)}</h3><span class="paper-summary-description">${esc(paper.summary)}</span></div>
        <span class="paper-summary-end"><span class="paper-status" data-state="${record.status}">${statuses[record.status]}</span><svg class="disclosure-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m8 10 4 4 4-4"/></svg></span>
      </summary>
      <div class="paper-content"><div class="paper-source-row"><div class="paper-bibliography"><p class="paper-full-title">${esc(paper.title)}</p><p class="paper-authors">${esc(paper.authors)} · ${esc(paper.venue)}</p></div>
        <div class="paper-source-links"><a class="paper-button" href="${esc(paper.url)}" target="_blank" rel="noopener noreferrer" aria-label="${esc(paper.shortTitle)}：查看原文，新标签页打开">查看原文 ${arrow}</a><a class="paper-button" href="${esc(paper.pdf)}" target="_blank" rel="noopener noreferrer" aria-label="${esc(paper.shortTitle)}：阅读 PDF，新标签页打开">阅读 PDF ${arrow}</a></div></div>
        <div class="reading-workspace"><div class="paper-guide"><h4>带着问题读</h4><ol>${paper.focus.map(question => `<li>${esc(question)}</li>`).join('')}</ol><div class="reading-cue"><strong>抓住这条主线</strong><p>${esc(paper.takeaway)}</p></div><div class="reading-cue"><strong>阅读边界</strong><p>${esc(paper.limitation)}</p></div></div>
          <div class="paper-notebook"><fieldset class="reading-state"><legend>${esc(paper.shortTitle)} · 阅读状态</legend><div class="reading-state-options">${Object.entries(statuses).map(([value, label]) => `<label><input type="radio" name="status-${paper.id}" value="${value}" data-status-id="${paper.id}"${record.status === value ? ' checked' : ''}><span>${label}</span></label>`).join('')}</div></fieldset>
            <div class="note-heading"><label for="note-${paper.id}">我的笔记<span class="sr-only">：${esc(paper.shortTitle)}</span></label><button class="insert-template" type="button" data-template-id="${paper.id}" aria-label="为${esc(paper.shortTitle)}插入笔记模板">插入模板</button></div>
            <textarea id="note-${paper.id}" data-note-id="${paper.id}" maxlength="20000" aria-describedby="save-${paper.id}" placeholder="这篇论文解决了什么？哪些证据说服了你？\n记下自己的理解与尚未解决的问题。">${esc(record.note)}</textarea><p class="note-save-state" id="save-${paper.id}">${storageAvailable ? (record.note ? '笔记已保存在本机' : '输入后自动保存到本机 · 最多 20,000 字符') : '本次页面暂存 · 请导出备份'}</p>
          </div></div></div></details>`;
  }

  byId('paper-list').innerHTML = papers.map(renderPaper).join('');
  const cards = new Map(papers.map(paper => [paper.id, byId('paper-' + paper.id)]));
  const searchText = new Map(papers.map(paper => [paper.id,
    [paper.title, paper.shortTitle, paper.authors, paper.year, paper.venue, paper.summary, categories[paper.category]].join(' ').toLocaleLowerCase()]));

  function updateProgress() {
    const done = papers.filter(paper => recordFor(paper.id).status === 'done').length;
    byId('completed-count').textContent = done;
    byId('total-count').textContent = papers.length;
    byId('reading-progress').max = papers.length;
    byId('reading-progress').value = done;
  }

  function filterPapers() {
    const query = byId('paper-search').value.trim().toLocaleLowerCase();
    const focusedCard = document.activeElement?.closest('.paper-card');
    let count = 0;
    for (const paper of papers) {
      const matches = (category === 'all' || paper.category === category)
        && (statusFilter === 'all' || recordFor(paper.id).status === statusFilter)
        && (!query || searchText.get(paper.id).includes(query));
      cards.get(paper.id).hidden = !matches;
      if (matches) count++;
    }
    byId('paper-count').textContent = count + ' 篇论文';
    byId('empty-papers').hidden = count !== 0;
    document.querySelectorAll('[data-category]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.category === category)));
    document.querySelectorAll('[data-status-filter]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.statusFilter === statusFilter)));
    if (focusedCard?.hidden) document.querySelector(`[data-status-filter="${statusFilter}"]`).focus();
  }

  function refreshRecords() {
    for (const paper of papers) {
      const record = recordFor(paper.id);
      const card = cards.get(paper.id);
      card.querySelectorAll('[data-status-id]').forEach(radio => { radio.checked = radio.value === record.status; });
      const badge = card.querySelector('.paper-status');
      badge.dataset.state = record.status;
      badge.textContent = statuses[record.status];
      const note = byId('note-' + paper.id);
      if (note.value !== record.note) note.value = record.note;
      byId('save-' + paper.id).textContent = storageAvailable
        ? (record.note ? '笔记已保存在本机' : '输入后自动保存到本机 · 最多 20,000 字符')
        : '本次页面暂存 · 请导出备份';
    }
    updateProgress();
    filterPapers();
  }

  byId('paper-search').addEventListener('input', filterPapers);
  document.querySelectorAll('[data-category]').forEach(button => button.addEventListener('click', () => {
    category = button.dataset.category;
    filterPapers();
  }));
  document.querySelectorAll('[data-status-filter]').forEach(button => button.addEventListener('click', () => {
    statusFilter = button.dataset.statusFilter;
    filterPapers();
  }));
  function resetFilters() {
    category = statusFilter = 'all';
    byId('paper-search').value = '';
    filterPapers();
  }
  byId('reset-filters').addEventListener('click', () => { resetFilters(); byId('paper-search').focus(); });

  function saveNote(input) {
    const id = input.dataset.noteId;
    commit({ [id]: { ...recordFor(id), note: input.value } });
    refreshRecords();
  }
  byId('paper-list').addEventListener('input', event => {
    if (event.target.matches('[data-note-id]')) saveNote(event.target);
  });
  byId('paper-list').addEventListener('change', event => {
    const id = event.target.dataset.statusId;
    if (!id) return;
    commit({ [id]: { ...recordFor(id), status: event.target.value } });
    refreshRecords();
  });
  byId('paper-list').addEventListener('click', event => {
    const button = event.target.closest('[data-template-id]');
    if (!button) return;
    const input = byId('note-' + button.dataset.templateId);
    const insertion = (input.value ? '\n\n' : '') + noteTemplate;
    if (input.value.length + insertion.length > input.maxLength) {
      notice('笔记已接近字数上限，请先整理内容再插入模板。', true);
      input.focus();
      return;
    }
    // Append so an existing note is never replaced by the template.
    input.setRangeText(insertion, input.value.length, input.value.length, 'end');
    saveNote(input);
    input.focus();
  });

  function openHash() {
    const id = new URLSearchParams(location.hash.slice(1)).get('paper');
    if (!paperIds.has(id)) return;
    resetFilters();
    const card = cards.get(id);
    card.open = true;
    requestAnimationFrame(() => card.scrollIntoView({ block: 'start', behavior: 'instant' }));
  }
  for (const [id, card] of cards) {
    card.addEventListener('toggle', () => {
      if (card.open && !card.hidden) {
        try { history.replaceState(null, '', '#paper=' + encodeURIComponent(id)); } catch { /* file:// histories may be restricted. */ }
      }
    });
  }
  window.addEventListener('hashchange', openHash);

  byId('export-papers').addEventListener('click', () => {
    const backup = { version: 1, exportedAt: new Date().toISOString(), records };
    const blob = new Blob([JSON.stringify(backup, null, 2)], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'study-hub-papers-' + new Date().toISOString().slice(0, 10) + '.json';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notice('已生成 JSON 备份，请在浏览器下载记录中确认保存。');
  });

  const dialog = byId('import-dialog');
  byId('choose-import').addEventListener('click', () => byId('import-papers').click());
  byId('import-papers').addEventListener('change', async event => {
    const sequence = ++importSequence;
    const file = event.target.files[0];
    event.target.value = '';
    if (!file) return;
    try {
      if (file.size > 1024 * 1024) throw new Error('备份文件不能超过 1 MiB。');
      const incoming = validateBackup(JSON.parse(await file.text()));
      if (sequence !== importSequence) return;
      const ids = Object.keys(incoming);
      if (!ids.length) { notice('这份备份没有阅读记录，当前笔记保持不变。'); return; }
      const conflicts = ids.filter(id => Object.hasOwn(records, id)
        && (records[id].note !== incoming[id].note || records[id].status !== incoming[id].status));
      pendingImport = incoming;
      byId('import-description').textContent = `将合并 ${ids.length} 篇论文的阅读记录。` + (conflicts.length
        ? `其中 ${conflicts.length} 篇与当前记录不同，确认后将用备份中的进度和笔记覆盖。`
        : '其余论文的进度和笔记保持不变。');
      dialog.returnValue = '';
      dialog.showModal();
      byId('cancel-import').focus();
    } catch (error) {
      pendingImport = null;
      notice('导入失败，当前记录保持不变：' + error.message, true);
    }
  });
  dialog.addEventListener('cancel', () => { dialog.returnValue = 'cancel'; });
  dialog.addEventListener('close', () => {
    const incoming = pendingImport;
    pendingImport = null;
    if (dialog.returnValue === 'confirm' && incoming) {
      commit(incoming);
      refreshRecords();
      notice(storageAvailable ? '已合并导入并保存到本机。' : '已在本次页面中合并导入，请及时导出备份。');
    }
    byId('choose-import').focus();
  });

  window.addEventListener('storage', event => {
    if (!storageAvailable || (event.key !== storageKey && event.key !== null)) return;
    try { records = readStored(); refreshRecords(); }
    catch { disableStorage('其他页面的记录无法读取，已停止自动写入。'); refreshRecords(); }
  });
  updateProgress();
  filterPapers();
  openHash();
  const sidebar = document.querySelector('.sidebar-nav');
  const active = sidebar.querySelector('[aria-current="page"]');
  if (sidebar.scrollWidth > sidebar.clientWidth && active) sidebar.scrollLeft = active.offsetLeft - sidebar.clientWidth / 2 + active.offsetWidth / 2;
})();
