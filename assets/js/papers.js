/* Paper library rendering, filters, navigation and read-only archive export. */
(() => {
  'use strict';
  const papers = window.STUDY_PAPERS;
  const categories = { ranking: '推荐排序', sequential: '序列建模', generative: '生成式推荐', architecture: '模型架构', tuning: '高效微调', alignment: '偏好对齐', agents: 'Agent 与检索' };
  const paperById = new Map(papers.map(paper => [paper.id, paper]));
  const lessons = window.PAPER_LESSONS;
  const learningPaths = window.PAPER_LEARNING_PATHS;
  const byId = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
  const searchInput = byId('paper-search');
  const paperList = byId('paper-list');
  const categoryButtons = [...document.querySelectorAll('[data-category]')];
  const collectionButtons = [...document.querySelectorAll('[data-collection]')];
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
  let category = 'all';
  let collection = 'translated';

  function notice(message, error = false) {
    const element = byId('paper-message');
    element.textContent = message;
    element.className = 'paper-notice' + (error ? ' error' : '');
    element.hidden = false;
  }

  const analysisFor = paper => window.PAPER_ANALYSES[paper.id];
  const collectionFor = paper => paper.collection || 'analysis';
  const sectionLink = (id, section) => '#paper=' + id + '&section=' + section;
  const externalLink = (url, label, className = '') => `<a${className ? ` class="${className}"` : ''} href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)} ${arrow}</a>`;

  function renderLesson(paper) {
    const lesson = lessons[paper.id];
    return `<section class="analysis-section paper-lesson" id="analysis-${paper.id}-study" tabindex="-1" aria-labelledby="lesson-title-${paper.id}">
      <div class="analysis-heading"><span aria-hidden="true">学</span><h4 id="lesson-title-${paper.id}">推导与动手学习</h4></div>
      <p class="lesson-label">以下算例、自测与实验方案为原创教学内容，非论文报告结果。</p>
      <div class="lesson-preparation"><p><strong>先备知识</strong>${esc(lesson.prerequisite)}</p><p><strong>学习目标</strong>${esc(lesson.goal)}</p></div>
      <div class="lesson-derivation"><h5>${esc(lesson.derivation.title)}</h5><ol>${lesson.derivation.steps.map(step => `<li>${esc(step)}</li>`).join('')}</ol><div class="analysis-formula"><code>${esc(lesson.derivation.formula)}</code></div></div>
      <aside class="analysis-example"><span class="example-label">手算与反例</span><p>${esc(lesson.worked)}</p></aside>
      ${lesson.code ? `<details class="lesson-code"><summary>查看最小 Python 算例<span>仅标准库 · 复制到本地运行</span></summary><pre tabindex="0" role="region" aria-label="${esc(paper.shortTitle)} Python 教学算例"><code>${esc(lesson.code)}</code></pre></details>` : ''}
      <div class="lesson-experiment"><h5>动手验证</h5><p>${esc(lesson.experiment.setup)}</p><ol>${lesson.experiment.steps.map(step => `<li>${esc(step)}</li>`).join('')}</ol><p class="lesson-check"><strong>如何验收：</strong>${esc(lesson.experiment.check)}</p></div>
      <div class="lesson-questions"><h5>先回答，再核对</h5>${lesson.questions.map((item,index) => `<div class="lesson-question"><h6>${index+1}. ${esc(item.q)}</h6><details><summary>查看参考答案<span class="sr-only">：${esc(item.q)}</span></summary><p>${esc(item.a)}</p></details></div>`).join('')}</div>
      <div class="lesson-related"><h5>带着问题接着读</h5>${lesson.related.map(([id,why]) => { const next = paperById.get(id); return `<p><a href="${sectionLink(id, 'study')}">${esc(next.shortTitle)} →</a><span>${esc(why)}</span></p>`; }).join('')}</div>
      <p class="lesson-source">方法依据：${externalLink(lesson.source.url, lesson.source.label)}</p>
    </section>`;
  }

  const pathContainer = byId('paper-learning-paths');
  pathContainer.innerHTML = `<details class="learning-path-disclosure"><summary><span><strong>按路线系统学习</strong><span>${learningPaths.length} 条路线 · ${papers.length} 篇进阶讲义 · ${Object.values(lessons).reduce((count, lesson) => count + lesson.questions.length, 0)} 道自测</span></span><svg class="disclosure-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m8 10 4 4 4-4"/></svg></summary><div class="paper-learning-routes">${learningPaths.map(route => `<section><h3>${esc(route.title)}</h3><p>${esc(route.description)}</p><ol>${route.ids.map(id => { const paper = paperById.get(id); return `<li><a href="#paper=${id}">${esc(paper.shortTitle)}</a></li>`; }).join('')}</ol><p class="route-outcome">${esc(route.outcome)}</p></section>`).join('')}</div></details>`;

  function renderAnalysis(paper) {
    if (paper.collection === 'translated') return `<article class="paper-analysis paper-translation-guide" aria-label="${esc(paper.shortTitle)}阅读导引">
      <div class="translation-heading"><div><span class="paper-eyebrow">中文全文 · 阅读导引</span><h4>${esc(paper.zhTitle)}</h4></div><span class="translation-file-info">${paper.pages} 页 · ${(paper.bytes / 1024 / 1024).toFixed(1)} MB</span></div>
      <nav class="analysis-nav" aria-label="${esc(paper.shortTitle)}学习目录"><a href="${sectionLink(paper.id, 'overview')}">阅读导引</a><a href="${sectionLink(paper.id, 'study')}">推导与练习</a></nav>
      <div class="translation-guide-sections" id="analysis-${paper.id}-overview" tabindex="-1">${[['problem', '01', '研究问题'], ['method', '02', '核心方法'], ['reading', '03', '精读线索']].map(([key, number, title]) => `<section><span>${number}</span><h5>${title}</h5><p>${esc(paper.guide[key])}</p></section>`).join('')}</div>
      ${renderLesson(paper)}
      <p class="translation-reading-note">先沿导引理解问题与方法，再打开中文全文查看公式、图表和实验细节。英文原文可用于对照术语与版本。</p>
    </article>`;
    const analysis = analysisFor(paper);
    if (!analysis) return `<div class="paper-analysis"><p class="paper-notice">${esc(paper.takeaway)} 详细解析未加载，请刷新页面重试。</p></div>`;
    const sectionId = section => 'analysis-' + paper.id + '-' + section;
    const heading = (number, title) => `<div class="analysis-heading"><span aria-hidden="true">${number}</span><h4>${title}</h4></div>`;
    return `<article class="paper-analysis" aria-label="${esc(paper.shortTitle)}中文解析">
      <div class="analysis-verdict"><span>先记住这个结论</span><p>${esc(analysis.verdict)}</p></div>
      <nav class="analysis-nav" aria-label="${esc(paper.shortTitle)}解析目录">${[
        ['overview', '研究问题'], ['method', '方法拆解'], ['evidence', '实验与证据'], ['limits', '贡献与边界'], ['study', '推导与练习'],
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
      ${renderLesson(paper)}
      <section class="analysis-section analysis-further">${heading('05', '想深入，再回到这几处原文')}
        <ul class="analysis-reading-path">${analysis.nextReading.map(item => `<li>${externalLink(item.url, item.label)}<p>${esc(item.detail)}</p></li>`).join('')}</ul>
        <div class="analysis-sources"><strong>资料来源</strong>${analysis.sources.map(source => externalLink(source.url, source.label)).join('')}</div>
      </section>
    </article>`;
  }

  function renderPaper(paper, index) {
    return `<details class="paper-card" id="paper-${paper.id}" data-paper-id="${paper.id}">
      <summary class="paper-summary"><span class="paper-index" aria-hidden="true">${String(index + 1).padStart(2, '0')}</span>
        <div class="paper-summary-copy"><span class="paper-meta"><span class="paper-category">${categories[paper.category]}</span><span class="paper-meta-dot" aria-hidden="true"></span><span>${paper.year}</span><span class="paper-analysis-badge">${paper.chinesePdf ? '中文全文' : '中文解析'} · 进阶讲义</span></span>
          <h3>${esc(paper.shortTitle)}</h3><span class="paper-summary-description">${esc(paper.summary)}</span></div>
        <span class="paper-summary-end"><svg class="disclosure-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m8 10 4 4 4-4"/></svg></span>
      </summary>
      <div class="paper-content"><div class="paper-source-row"><div class="paper-bibliography"><p class="paper-full-title">${esc(paper.title)}</p><p class="paper-authors">${esc(paper.authors)} · ${esc(paper.venue)}</p></div>
        <div class="paper-source-links">${paper.chinesePdf ? `<a class="paper-button primary chinese-pdf-link" href="${esc(paper.chinesePdf)}" target="_blank" rel="noopener" aria-label="${esc(paper.shortTitle)}：阅读中文 PDF，新标签页打开">阅读中文 PDF ${arrow}</a>${externalLink(paper.url, '英文原文', 'paper-button')}` : `${externalLink(paper.url, '英文原文', 'paper-button')}${externalLink(paper.pdf, '英文 PDF', 'paper-button')}`}</div></div>
        <div class="reading-workspace">${renderAnalysis(paper)}
        </div></div></details>`;
  }

  paperList.innerHTML = papers.map(renderPaper).join('');
  const cards = new Map(papers.map(paper => [paper.id, byId('paper-' + paper.id)]));
  const searchText = new Map(papers.map(paper => [paper.id,
    [paper.title, paper.zhTitle || '', paper.shortTitle, paper.authors, paper.year, paper.venue, paper.summary, categories[paper.category],
      cards.get(paper.id).querySelector('.paper-analysis').textContent,
    ].join(' ').toLocaleLowerCase()]));

  function filterPapers() {
    const query = searchInput.value.trim().toLocaleLowerCase();
    const focusedCard = document.activeElement?.closest('.paper-card');
    let count = 0;
    for (const paper of papers) {
      const matches = (collection === 'all' || collectionFor(paper) === collection)
        && (category === 'all' || paper.category === category)
        && (!query || searchText.get(paper.id).includes(query));
      const card = cards.get(paper.id);
      const hidden = !matches;
      if (card.hidden !== hidden) card.hidden = hidden;
      if (matches) count++;
    }
    byId('paper-count').textContent = count + ' 篇论文';
    byId('empty-papers').hidden = count !== 0;
    const filterCount = Number(category !== 'all');
    byId('active-filter-count').textContent = filterCount;
    byId('active-filter-count').hidden = !filterCount;
    byId('filter-description').textContent = (categories[category] || '全部主题');
    categoryButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.category === category)));
    collectionButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.collection === collection)));
    if (focusedCard?.hidden) byId('paper-filters').querySelector('summary').focus();
  }

  searchInput.addEventListener('input', filterPapers);
  collectionButtons.forEach(button => button.addEventListener('click', () => {
    closePanels();
    collection = button.dataset.collection;
    category = 'all';
    searchInput.value = '';
    filterPapers();
  }));
  categoryButtons.forEach(button => button.addEventListener('click', () => {
    category = button.dataset.category;
    filterPapers();
  }));
  function resetFilters() {
    collection = category = 'all';
    searchInput.value = '';
    filterPapers();
  }
  byId('reset-filters').addEventListener('click', () => { resetFilters(); searchInput.focus(); });

  function openHash() {
    const route = new URLSearchParams(location.hash.slice(1));
    const id = route.get('paper');
    if (!paperById.has(id)) return false;
    const card = cards.get(id);
    if (card.hidden) resetFilters();
    card.open = true;
    const section = route.get('section');
    const target = ['overview', 'method', 'evidence', 'limits', 'study'].includes(section)
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
  pathContainer.addEventListener('click', event => {
    const link = event.target.closest('a');
    if (link && link.hash === location.hash) { event.preventDefault(); openHash(); }
  });
  paperList.addEventListener('click', event => {
    const link = event.target.closest('.analysis-nav a, .lesson-related a');
    if (link && link.hash === location.hash) { event.preventDefault(); openHash(); }
  });

  byId('export-papers').addEventListener('click', () => {
    // Read at click time: preserve every stored byte without migration or writes.
    let archive;
    try { archive = localStorage.getItem('study-hub:papers:v1'); }
    catch { notice('无法读取浏览器中的历史笔记，请检查存储权限。', true); return; }
    if (!archive) { notice('当前浏览器中没有历史笔记。'); return; }
    const url = URL.createObjectURL(new Blob([archive], { type: 'application/json;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'study-hub-papers-' + new Date().toISOString().slice(0, 10) + '.json';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notice('已导出历史笔记，浏览器中的原始记录保持不变。');
  });

  filterPapers();
  openHash();
})();
