/* Local paper notebook. Plain scripts also work when opened with file://. */
(() => {
  'use strict';
  window.StudyHubNavigation?.mountAll();
  const papers = window.STUDY_PAPERS;
  const storageKey = 'study-hub:papers:v1';
  const categories = { ranking: '推荐排序', sequential: '序列建模', generative: '生成式推荐', architecture: '模型架构', tuning: '高效微调', alignment: '偏好对齐', agents: 'Agent 与检索' };
  const statuses = { unread: '未开始', reading: '在读', done: '已读' };
  const paperIds = new Set(papers.map(paper => paper.id));
  const byId = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
  const arrow = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M7 7h10v10"/></svg>';
  const disclosures = ['paper-filters', 'paper-tools'].map(byId);
  function closePanels() { disclosures.forEach(panel => { panel.open = false; }); }
  function placePanel(disclosure) {
    const panel = disclosure.querySelector('.paper-popover');
    const trigger = disclosure.querySelector('summary').getBoundingClientRect();
    const width = Math.min(420, window.innerWidth - 32);
    panel.style.width = width + 'px';
    panel.style.maxHeight = 'none';
    const desired = panel.scrollHeight;
    const below = window.innerHeight - trigger.bottom - 16;
    const above = trigger.top - 16;
    const upwards = below < Math.min(desired, 280) && above > below;
    const height = Math.min(desired, Math.max(80, (upwards ? above : below) - 8));
    panel.style.maxHeight = height + 'px';
    panel.style.left = Math.max(16, Math.min(trigger.right - width, window.innerWidth - width - 16)) + 'px';
    panel.style.top = Math.max(8, upwards ? trigger.top - height - 8 : trigger.bottom + 8) + 'px';
  }
  disclosures.forEach(disclosure => {
    disclosure.addEventListener('toggle', () => {
      if (!disclosure.open) return;
      disclosures.forEach(other => { if (other !== disclosure) other.open = false; });
      placePanel(disclosure);
    });
    disclosure.addEventListener('focusout', event => {
      if (event.relatedTarget) {
        if (!disclosure.contains(event.relatedTarget)) disclosure.open = false;
        return;
      }
      // During pointer focus changes activeElement can briefly be body.
      setTimeout(() => {
        if (disclosure.open && !disclosure.contains(document.activeElement)) disclosure.open = false;
      }, 0);
    });
  });
  document.addEventListener('pointerdown', event => {
    disclosures.forEach(disclosure => { if (!disclosure.contains(event.target)) disclosure.open = false; });
  });
  document.addEventListener('keydown', event => {
    const opened = disclosures.find(disclosure => disclosure.open);
    if (event.key === 'Escape' && opened) {
      event.preventDefault();
      opened.open = false;
      opened.querySelector('summary').focus();
    }
  });
  window.addEventListener('resize', () => disclosures.filter(panel => panel.open).forEach(placePanel));
  window.addEventListener('scroll', event => {
    if (!disclosures.some(disclosure => disclosure.contains(event.target))) closePanels();
  }, { passive: true, capture: true });
  const noteTemplate = '## 研究问题\n作者想解决什么？已有方法的不足是什么？\n\n## 核心方法\n用自己的话说明关键步骤。\n\n## 实验证据\n记录数据、基线、评价指标和关键消融。\n\n## 局限与疑问\n哪些结论还需要验证？\n\n## 我的理解\n用一句话总结，并联系已有知识。\n';
  let records = Object.create(null);
  let storageAvailable = true;
  let category = 'all';
  let collection = 'translated';
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

  const analysisFor = paper => window.PAPER_ANALYSES?.[paper.id];
  const collectionFor = paper => paper.collection || 'analysis';
  const sectionLink = (id, section) => '#paper=' + id + '&section=' + section;
  const externalLink = (url, label, className = '') => `<a${className ? ` class="${className}"` : ''} href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)} ${arrow}</a>`;

  function renderAnalysis(paper) {
    if (paper.collection === 'translated') return `<article class="paper-analysis paper-translation-guide" aria-label="${esc(paper.shortTitle)}阅读导引">
      <div class="translation-heading"><div><span class="paper-eyebrow">中文全文 · 阅读导引</span><h4>${esc(paper.zhTitle)}</h4></div><span class="translation-file-info">${paper.pages} 页 · ${(paper.bytes / 1024 / 1024).toFixed(1)} MB</span></div>
      <div class="translation-guide-sections">${[['problem', '01', '研究问题'], ['method', '02', '核心方法'], ['reading', '03', '精读线索']].map(([key, number, title]) => `<section><span>${number}</span><h5>${title}</h5><p>${esc(paper.guide[key])}</p></section>`).join('')}</div>
      <p class="translation-reading-note">先沿导引理解问题与方法，再打开中文全文查看公式、图表和实验细节。英文原文可用于对照术语与版本。</p>
    </article>`;
    const analysis = analysisFor(paper);
    if (!analysis) return `<div class="paper-analysis"><p class="paper-notice">${esc(paper.takeaway)} 详细解析未加载，请刷新页面重试。</p></div>`;
    const sectionId = section => 'analysis-' + paper.id + '-' + section;
    const heading = (number, title) => `<div class="analysis-heading"><span aria-hidden="true">${number}</span><h4>${title}</h4></div>`;
    return `<article class="paper-analysis" aria-label="${esc(paper.shortTitle)}中文解析">
      <div class="analysis-verdict"><span>先记住这个结论</span><p>${esc(analysis.verdict)}</p></div>
      <nav class="analysis-nav" aria-label="${esc(paper.shortTitle)}解析目录">${[
        ['overview', '研究问题'], ['method', '方法拆解'], ['evidence', '实验与证据'], ['limits', '贡献与边界'], ['notes', '我的笔记'],
      ].map(([section, label]) => `<a href="${sectionLink(paper.id, section)}">${label}</a>`).join('')}</nav>
      <section class="analysis-section" id="${sectionId('overview')}" tabindex="-1">${heading('01', '为什么需要这项研究')}
        <div class="analysis-comparison"><div><h5>研究问题</h5><p>${esc(analysis.problem)}</p></div><div><h5>已有方法的不足</h5><p>${esc(analysis.baseline)}</p></div></div>
      </section>
      <section class="analysis-section" id="${sectionId('method')}" tabindex="-1">${heading('02', '方法是怎么工作的')}
        <ol class="analysis-flow" style="--flow-count:${analysis.flow.length}" aria-label="方法流程">${analysis.flow.map((step, index) => `<li><span class="flow-step-number">${String(index + 1).padStart(2, '0')}</span><strong>${esc(step.label)}</strong><p>${esc(step.detail)}</p></li>`).join('')}</ol>
        <div class="analysis-method">${analysis.method.map(step => `<section><h5>${esc(step.title)}</h5><p>${esc(step.text)}</p></section>`).join('')}</div>
        <div class="analysis-formula"><h5>抓住核心关系</h5><code>${esc(analysis.formula.expression)}</code><p>${esc(analysis.formula.explanation)}</p></div>
        <aside class="analysis-example"><span class="example-label">理解示例 · 非论文实验</span><h5>${esc(analysis.example.title)}</h5><p>${esc(analysis.example.text)}</p></aside>
      </section>
      <section class="analysis-section" id="${sectionId('evidence')}" tabindex="-1">${heading('03', '实验究竟证明了什么')}
        <p class="analysis-section-intro">${esc(analysis.evidence.intro)}</p>
        <p class="analysis-table-hint">左右滑动，查看完整结果与比较条件 →</p>
        <div class="analysis-table-scroll" role="region" aria-label="${esc(paper.shortTitle)}实验结果，可横向滚动" tabindex="0"><table class="analysis-evidence"><caption class="sr-only">论文结果、比较条件与解释</caption><thead><tr><th scope="col">比较项目</th><th scope="col">论文结果</th><th scope="col">条件与解读</th></tr></thead><tbody>${analysis.evidence.items.map(item => `<tr><th scope="row">${esc(item.label)}</th><td><strong>${esc(item.value)}</strong></td><td><p>${esc(item.detail)}</p>${externalLink(item.url, '核对原文', 'evidence-source')}</td></tr>`).join('')}</tbody></table></div>
        <div class="analysis-evidence-conclusion"><strong>如何理解这些证据</strong><p>${esc(analysis.evidence.takeaway)}</p></div>
      </section>
      <section class="analysis-section" id="${sectionId('limits')}" tabindex="-1">${heading('04', '贡献与结论的边界')}
        <div class="analysis-comparison"><div><h5>值得记住的贡献</h5><ul>${analysis.contributions.map(value => `<li>${esc(value)}</li>`).join('')}</ul></div><div><h5>不能直接推导出的结论</h5><ul>${analysis.limits.map(value => `<li>${esc(value)}</li>`).join('')}</ul></div></div>
        <div class="analysis-application"><h5>应用判断</h5><p><strong>可以借鉴：</strong>${esc(analysis.application.fit)}</p><p><strong>还需验证：</strong>${esc(analysis.application.caution)}</p></div>
      </section>
      <section class="analysis-section analysis-further">${heading('05', '想深入，再回到这几处原文')}
        <ul class="analysis-reading-path">${analysis.nextReading.map(item => `<li>${externalLink(item.url, item.label)}<p>${esc(item.detail)}</p></li>`).join('')}</ul>
        <div class="analysis-sources"><strong>资料来源</strong>${analysis.sources.map(source => externalLink(source.url, source.label)).join('')}</div>
      </section>
    </article>`;
  }

  function renderPaper(paper, index) {
    const record = recordFor(paper.id);
    return `<details class="paper-card" id="paper-${paper.id}" data-paper-id="${paper.id}">
      <summary class="paper-summary"><span class="paper-index" aria-hidden="true">${String(index + 1).padStart(2, '0')}</span>
        <div class="paper-summary-copy"><span class="paper-meta"><span class="paper-category">${categories[paper.category]}</span><span class="paper-meta-dot" aria-hidden="true"></span><span>${paper.year}</span><span class="paper-analysis-badge">${paper.chinesePdf ? '中文全文' : '中文解析'}</span></span>
          <h3>${esc(paper.shortTitle)}</h3><span class="paper-summary-description">${esc(paper.summary)}</span></div>
        <span class="paper-summary-end"><span class="paper-status" data-state="${record.status}">${statuses[record.status]}</span><svg class="disclosure-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m8 10 4 4 4-4"/></svg></span>
      </summary>
      <div class="paper-content"><div class="paper-source-row"><div class="paper-bibliography"><p class="paper-full-title">${esc(paper.title)}</p><p class="paper-authors">${esc(paper.authors)} · ${esc(paper.venue)}</p></div>
        <div class="paper-source-links">${paper.chinesePdf ? `<a class="paper-button primary chinese-pdf-link" href="${esc(paper.chinesePdf)}" target="_blank" rel="noopener" aria-label="${esc(paper.shortTitle)}：阅读中文 PDF，新标签页打开">阅读中文 PDF ${arrow}</a>${externalLink(paper.url, '英文原文', 'paper-button')}` : `${externalLink(paper.url, '英文原文', 'paper-button')}${externalLink(paper.pdf, '英文 PDF', 'paper-button')}`}</div></div>
        <div class="reading-workspace">${renderAnalysis(paper)}
          <div class="paper-notebook" id="analysis-${paper.id}-notes" tabindex="-1"><div class="notebook-progress"><h4>留下自己的理解</h4><p>把最有启发的结论、还没想清楚的问题，写成自己的话。</p><fieldset class="reading-state"><legend>${esc(paper.shortTitle)} · 阅读状态</legend><div class="reading-state-options">${Object.entries(statuses).map(([value, label]) => `<label><input type="radio" name="status-${paper.id}" value="${value}" data-status-id="${paper.id}"${record.status === value ? ' checked' : ''}><span>${label}</span></label>`).join('')}</div></fieldset></div>
            <div class="notebook-editor"><div class="note-heading"><label for="note-${paper.id}">我的笔记<span class="sr-only">：${esc(paper.shortTitle)}</span></label><button class="insert-template" type="button" data-template-id="${paper.id}" aria-label="为${esc(paper.shortTitle)}插入笔记模板">插入模板</button></div>
            <textarea id="note-${paper.id}" data-note-id="${paper.id}" maxlength="20000" aria-describedby="save-${paper.id}" placeholder="这篇论文解决了什么？哪些证据说服了你？\n记下自己的理解与尚未解决的问题。">${esc(record.note)}</textarea><p class="note-save-state" id="save-${paper.id}">${storageAvailable ? (record.note ? '笔记已保存在本机' : '输入后自动保存到本机 · 最多 20,000 字符') : '本次页面暂存 · 请导出备份'}</p>
          </div></div></div></div></details>`;
  }

  byId('paper-list').innerHTML = papers.map(renderPaper).join('');
  const cards = new Map(papers.map(paper => [paper.id, byId('paper-' + paper.id)]));
  const searchText = new Map(papers.map(paper => [paper.id,
    [paper.title, paper.zhTitle || '', paper.shortTitle, paper.authors, paper.year, paper.venue, paper.summary, categories[paper.category],
      cards.get(paper.id).querySelector('.paper-analysis').textContent,
    ].join(' ').toLocaleLowerCase()]));

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
      const matches = (collection === 'all' || collectionFor(paper) === collection)
        && (category === 'all' || paper.category === category)
        && (statusFilter === 'all' || recordFor(paper.id).status === statusFilter)
        && (!query || searchText.get(paper.id).includes(query));
      cards.get(paper.id).hidden = !matches;
      if (matches) count++;
    }
    byId('paper-count').textContent = count + ' 篇论文';
    byId('empty-papers').hidden = count !== 0;
    const filterCount = Number(category !== 'all') + Number(statusFilter !== 'all');
    byId('active-filter-count').textContent = filterCount;
    byId('active-filter-count').hidden = !filterCount;
    byId('filter-description').textContent = (categories[category] || '全部主题') + ' · ' + (statusFilter === 'all' ? '全部状态' : statuses[statusFilter]);
    document.querySelectorAll('[data-category]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.category === category)));
    document.querySelectorAll('[data-status-filter]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.statusFilter === statusFilter)));
    document.querySelectorAll('[data-collection]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.collection === collection)));
    if (focusedCard?.hidden) byId('paper-filters').querySelector('summary').focus();
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
  document.querySelectorAll('[data-collection]').forEach(button => button.addEventListener('click', () => {
    closePanels();
    collection = button.dataset.collection;
    category = statusFilter = 'all';
    byId('paper-search').value = '';
    filterPapers();
  }));
  document.querySelectorAll('[data-category]').forEach(button => button.addEventListener('click', () => {
    category = button.dataset.category;
    filterPapers();
  }));
  document.querySelectorAll('[data-status-filter]').forEach(button => button.addEventListener('click', () => {
    statusFilter = button.dataset.statusFilter;
    filterPapers();
  }));
  function resetFilters() {
    collection = category = statusFilter = 'all';
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
    const route = new URLSearchParams(location.hash.slice(1));
    const id = route.get('paper');
    if (!paperIds.has(id)) return false;
    const card = cards.get(id);
    if (card.hidden) resetFilters();
    card.open = true;
    const section = route.get('section');
    const target = ['overview', 'method', 'evidence', 'limits', 'notes'].includes(section)
      ? byId('analysis-' + id + '-' + section) || card : card;
    requestAnimationFrame(() => {
      target.scrollIntoView({ block: 'start', behavior: 'instant' });
      (target === card ? card.querySelector('summary') : target).focus({ preventScroll: true });
    });
    return true;
  }
  for (const [id, card] of cards) {
    card.addEventListener('toggle', () => {
      if (card.open && !card.hidden && new URLSearchParams(location.hash.slice(1)).get('paper') !== id) {
        try { history.replaceState(null, '', '#paper=' + encodeURIComponent(id)); } catch { /* file:// histories may be restricted. */ }
      }
    });
  }
  window.addEventListener('hashchange', openHash);
  byId('paper-list').addEventListener('click', event => {
    const link = event.target.closest('.analysis-nav a');
    if (link && link.hash === location.hash) { event.preventDefault(); openHash(); }
  });

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
  byId('choose-import').addEventListener('click', () => {
    closePanels();
    byId('paper-tools').querySelector('summary').focus();
    byId('import-papers').click();
  });
  byId('import-papers').addEventListener('change', async event => {
    const sequence = ++importSequence;
    const file = event.target.files[0];
    event.target.value = '';
    if (!file) return;
    try {
      if (file.size > 8 * 1024 * 1024) throw new Error('备份文件不能超过 8 MiB。');
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
    byId('paper-tools').querySelector('summary').focus();
  });

  window.addEventListener('storage', event => {
    if (!storageAvailable || (event.key !== storageKey && event.key !== null)) return;
    try { records = readStored(); refreshRecords(); }
    catch { disableStorage('其他页面的记录无法读取，已停止自动写入。'); refreshRecords(); }
  });
  updateProgress();
  filterPapers();
  openHash();
})();
