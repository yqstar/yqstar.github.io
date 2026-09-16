/* Shared, local-file-friendly navigation for the overview and reading pages. */
(() => {
  const icons = {
    overview: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    algorithms: '<path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-13-2 16"/>',
    interviews: '<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
    papers: '<path d="M12 5c-3-2-6-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1Zm0 0v15"/>',
  };
  const escape = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
  function markup(root = './', active = 'overview') {
    const prefix = root.endsWith('/') ? root : root + '/';
    const links = [
      ['overview', 'index.html', '学习概览', '概览'],
      ['algorithms', 'pages/algorithms.html', '算法训练场', '算法'],
      ['interviews', 'index.html#interviews', 'AI 专题', 'AI 专题'],
      ['papers', 'pages/papers.html', '论文阅读', '论文'],
    ];
    return `<header class="site-sidebar"><a class="brand" href="${escape(prefix)}index.html" aria-label="Study Hub 首页"><span class="brand-mark" aria-hidden="true">s<span>_</span></span><span class="brand-wordmark">Study Hub</span></a><nav class="sidebar-nav" aria-label="学习空间导航">${links.map(([key, path, label, short]) => `<a class="sidebar-link" href="${escape(prefix + path)}" aria-label="${label}"${key === active ? ' aria-current="page"' : ''}><svg viewBox="0 0 24 24" aria-hidden="true">${icons[key]}</svg><span class="nav-text">${label}</span><span class="nav-short" aria-hidden="true">${short}</span></a>`).join('')}</nav></header>`;
  }
  function mountAll() {
    document.querySelectorAll('[data-study-navigation]').forEach(slot => {
      slot.outerHTML = markup(slot.dataset.root || './', slot.dataset.active || 'overview');
    });
  }
  window.StudyHubNavigation = { markup, mountAll };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountAll, { once: true });
  else mountAll();
})();
