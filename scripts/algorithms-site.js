// Site presentation only. Keep upstream controls, records and runtime behavior intact.
(() => {
  const style = document.createElement("style");
  style.id = "study-hub-navigation-style";
  style.textContent = `
    :root { --header-height: 76px; }
    .app-header { gap: 24px; height: var(--header-height); padding: 12px max(32px,calc((100% - 1180px)/2)); background: var(--surface); border-bottom: 1px solid var(--line); box-shadow: none; backdrop-filter: none; }
    #study-hub-home { display: inline-flex; flex: 0 0 auto; align-items: center; gap: 10px; color: var(--text); font-size: 18px; font-weight: 650; letter-spacing: -.4px; text-decoration: none; white-space: nowrap; }
    #study-hub-home .brand-mark { display: grid; place-content: center; grid-auto-flow: column; width: 32px; height: 32px; flex-shrink: 0; background: var(--text); color: var(--surface); border: 1px solid transparent; border-radius: 10px; box-shadow: none; font: 700 21px/1 "SFMono-Regular",Consolas,"Liberation Mono",monospace; letter-spacing: -3px; padding-right: 3px; }
    #study-hub-home .brand-mark span { opacity: .65; }
    .app-header .sidebar-nav { display: flex; align-items: center; gap: 6px; min-width: 0; margin-left: auto; }
    .app-header .sidebar-link { display: inline-flex; align-items: center; justify-content: center; gap: 7px; min-height: 42px; padding: 10px 14px; border: 1px solid transparent; border-radius: 9px; color: var(--muted); font-size: 13px; line-height: 1.5; text-decoration: none; white-space: nowrap; }
    .app-header .sidebar-link svg { width: 17px; height: 17px; flex-shrink: 0; fill: none; stroke: currentColor; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
    .app-header .sidebar-link:hover { color: var(--text); background: var(--surface-2); }
    .app-header .sidebar-link:active { background: var(--line); }
    .app-header .sidebar-link[aria-current="page"] { color: var(--accent); background: var(--accent-soft); font-weight: 550; }
    .app-header .nav-short { display: none; }
    .app-header .header-actions { width: auto; flex: 0 0 auto; gap: 8px; }
    #theme-button { width: 44px; height: 44px; flex: 0 0 auto; padding: 10px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); color: var(--text); }
    #theme-button:hover { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }
    #study-hub-tools { position: relative; }
    #study-hub-tools > summary { display: flex; align-items: center; justify-content: center; gap: 8px; min-height: 44px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; color: var(--text); background: var(--surface); font-size: 13px; font-weight: 500; line-height: 1.3; list-style: none; cursor: pointer; user-select: none; }
    #study-hub-tools > summary::-webkit-details-marker, .practice-stats > summary::-webkit-details-marker { display: none; }
    #study-hub-tools > summary:hover { background: var(--surface-2); border-color: var(--line-strong); }
    #study-hub-tools > summary:active { background: var(--surface-3); }
    #study-hub-tools[open] > summary { color: var(--accent); border-color: var(--accent); }
    .site-disclosure-chevron { width: 12px; height: 12px; flex: 0 0 auto; fill: none; stroke: currentColor; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
    details[open] > summary > .site-disclosure-chevron { transform: rotate(180deg); }
    .study-tools-panel { position: absolute; z-index: 50; top: calc(100% + 8px); right: 0; width: 296px; max-width: calc(100vw - 24px); max-height: calc(100dvh - var(--header-height) - 20px); overflow-y: auto; overscroll-behavior: contain; padding: 8px; border: 1px solid var(--line-strong); border-radius: 16px; color: var(--text); background: var(--surface); box-shadow: var(--shadow); }
    .study-tools-section { padding: 8px; }
    .study-tools-section + .study-tools-section { border-top: 1px solid var(--line); }
    .study-tools-section h2 { margin: 0 0 8px; padding: 0 4px; color: var(--muted); font-size: 11px; font-weight: 600; line-height: 1.5; }
    .study-tools-panel .header-action-label { display: inline; }
    .study-tools-panel .button, .study-tools-panel .icon-button { width: auto; height: auto; min-height: 40px; padding: 8px 10px; font-size: 13px; }
    .study-tools-panel #auto-expand-button { display: flex; justify-content: space-between; width: 100%; padding: 8px 4px; }
    .study-tools-mode { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 48px; padding-left: 4px; font-size: 13px; }
    .study-tools-panel .code-mode-switch { height: 36px; }
    .study-tools-panel .code-mode-button { min-width: 52px; padding: 6px 10px; }
    .study-tools-panel .header-data-actions { display: grid; grid-template-columns: 1fr 1fr; }
    .study-tools-panel .header-data-actions .button { border-radius: 0; }
    .study-tools-panel #undo-import-button { width: 100%; margin-top: 6px; text-align: left; }
    .study-tools-panel #reset-progress-button { width: 100%; margin-top: 6px; justify-content: flex-start; text-align: left; }
    .practice-overview { margin-bottom: 24px; border-radius: 18px; background: var(--surface); box-shadow: none; }
    .practice-overview .hero { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 24px; padding: 24px 28px; }
    .practice-overview .hero-copy { display: block; }
    .practice-overview .eyebrow { display: none; }
    .practice-overview .hero h1 { max-width: none; font-size: 28px; line-height: 1.3; }
    .practice-overview .hero-description { max-width: none; margin: 8px 0 0; font-size: 13px; line-height: 1.6; }
    .practice-overview .hero-actions { display: flex; gap: 8px; align-items: center; }
    .practice-overview .hero-actions .button { padding-inline: 16px; border-radius: 10px; font-size: 13px; }
    .practice-overview .hero-actions .primary::after { display: none; }
    .practice-stats { border-top: 1px solid var(--line); }
    .practice-stats > summary { display: flex; align-items: center; gap: 10px; min-height: 44px; padding: 10px 28px; color: var(--muted); font-size: 12px; line-height: 1.5; list-style: none; cursor: pointer; }
    .practice-stats > summary:hover { color: var(--text); background: var(--surface-2); }
    .practice-stats > summary > .site-disclosure-chevron { margin-left: auto; }
    .practice-stats-hint { color: var(--faint); font-size: 11px; }
    .practice-stats-content { display: grid; grid-template-columns: minmax(260px, 1fr) 1.6fr; gap: 12px 28px; padding: 12px 28px 20px; }
    .practice-stats .hero-progress { display: flex; grid-column: auto; grid-row: auto; padding: 0; gap: 20px; }
    .practice-stats .progress-orbit { width: 88px; height: 88px; }
    .practice-stats .progress-number { font-size: 26px; }
    .practice-stats .progress-detail { display: block; }
    .practice-stats .progress-detail h2 { display: none; }
    .practice-stats .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); align-content: center; padding: 0; border: 0; background: transparent; }
    .practice-stats .summary-card { display: flex; padding: 10px 16px; }
    .practice-stats .summary-card:nth-child(3) { border-left: 0; }
    .practice-stats .summary-card:nth-child(n+3) { border-top: 1px solid var(--line); }
    .practice-stats .summary-value { margin-top: 0; font-size: 22px; }
    .practice-stats .practice-caption { grid-column: 1 / -1; margin: 0; }
    @media (max-width: 980px) {
      .practice-overview .hero { grid-template-columns: minmax(0, 1fr); gap: 18px; }
    }
    @media (max-width: 640px) {
      .practice-overview .hero { padding: 20px; gap: 16px; }
      .practice-overview .hero h1 { font-size: 23px; letter-spacing: -.025em; }
      .practice-overview .hero-actions .button { flex: 1; padding-inline: 8px; min-height: 42px; font-size: 12px; }
      .practice-stats > summary { padding-inline: 20px; }
      .practice-stats-content { grid-template-columns: minmax(0, 1fr); gap: 16px; padding: 8px 20px 18px; }
      .practice-stats .hero-progress { justify-content: flex-start; }
      .practice-stats .summary-card { padding: 10px 8px; gap: 8px; }
      .practice-stats .summary-label { font-size: 11px; }
      .practice-stats .summary-label .ui-icon { display: none; }
      .practice-stats .summary-value { font-size: 20px; }
    }
    @media (max-width: 860px) {
      :root { --header-height: 113px; }
      .app-header { flex-wrap: wrap; gap: 10px; padding: 14px 20px 10px; }
      #study-hub-home { font-size: 17px; }
      #study-hub-home .brand-mark { width: 29px; height: 29px; font-size: 19px; border-radius: 9px; }
      .app-header .header-actions { margin-left: auto; gap: 8px; }
      .app-header .sidebar-nav { order: 3; width: 100%; display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 4px; margin-left: 0; }
      .app-header .sidebar-link { padding: 8px 6px; min-height: 40px; font-size: 12px; }
      .app-header .sidebar-link svg,.app-header .nav-text { display: none; }
      .app-header .nav-short { display: inline; }
      #theme-button { width: 38px; height: 38px; min-height: 38px; padding: 9px; }
      #study-hub-tools > summary { min-height: 38px; padding: 8px 10px; font-size: 12px; }
      .study-tools-panel { position: fixed; top: calc(var(--header-height) + 4px); right: 16px; }
    }
    @media (max-width: 380px) { .app-header { padding-inline: 16px; } }
    @media (prefers-reduced-transparency: reduce) { .study-tools-panel { background: var(--surface); } }
    @media (prefers-contrast: more) { .study-tools-panel { border-color: var(--text); background: var(--surface); } }
  `;
  document.head.append(style);

  const chevron = '<svg class="site-disclosure-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4"></path></svg>';
  const header = document.getElementById("app-header");
  // Reuse the site's real brand and destinations; embed the shared renderer at build time.
  const template = document.createElement("template");
  template.innerHTML = window.StudyHubNavigation.markup("../", "algorithms");
  const homeLink = template.content.querySelector(".brand");
  homeLink.id = "study-hub-home";
  const navigation = template.content.querySelector(".sidebar-nav");
  header.querySelector(".header-leading").replaceWith(homeLink, navigation);

  // Move the existing controls rather than replacing their state or event handlers.
  const actions = header.querySelector(".header-actions");
  const tools = document.createElement("details");
  tools.id = "study-hub-tools";
  tools.innerHTML = '<summary aria-controls="study-tools-panel">工具' + chevron + '</summary><div id="study-tools-panel" class="study-tools-panel" role="group" aria-label="练习工具"><section class="study-tools-section" aria-labelledby="study-settings-title"><h2 id="study-settings-title">练习设置</h2><div class="study-tools-mode"><span>代码模式</span></div></section><section class="study-tools-section" aria-labelledby="study-records-title"><h2 id="study-records-title">学习记录</h2></section></div>';
  const preferences = tools.querySelector(".study-tools-section");
  const mode = tools.querySelector(".study-tools-mode");
  const records = tools.querySelectorAll(".study-tools-section")[1];
  preferences.insertBefore(document.getElementById("auto-expand-button"), mode);
  preferences.querySelector(".header-action-label").textContent = "默认展开题面";
  mode.append(header.querySelector(".code-mode-switch"));
  records.append(header.querySelector(".header-data-actions"), document.getElementById("undo-import-button"), document.getElementById("reset-progress-button"));
  records.querySelector("#undo-import-button").textContent = "↶ 撤销上次导入";
  const theme = document.getElementById("theme-button");
  actions.replaceChildren(tools, theme);

  const toolsSummary = tools.querySelector("summary");
  let toolDialog = null;
  function closeTools(returnFocus = false) {
    if (!tools.open) return;
    tools.open = false;
    if (returnFocus) toolsSummary.focus({ preventScroll: true });
  }
  document.addEventListener("pointerdown", event => {
    if (!tools.contains(event.target)) closeTools();
  });
  document.addEventListener("focusin", event => {
    if (!tools.contains(event.target)) closeTools();
  });
  document.addEventListener("keydown", event => {
    if (event.key !== "Escape" || !tools.open) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    closeTools(true);
  }, true);
  tools.addEventListener("click", event => {
    const button = event.target.closest("button");
    const dialogs = { "export-button": "export-modal", "import-button": "import-modal", "reset-progress-button": "reset-progress-modal" };
    if (!button || !dialogs[button.id]) return;
    toolDialog = document.getElementById(dialogs[button.id]);
    closeTools(true);
  }, true);
  // Upstream dialogs restore focus to their trigger; that trigger is now folded.
  // Return to the visible disclosure after the original dialog has closed.
  for (const id of ["export-modal", "import-modal", "reset-progress-modal"]) {
    const modal = document.getElementById(id);
    let wasOpen = false;
    new MutationObserver(() => {
      const open = !modal.classList.contains("hidden");
      if (wasOpen && !open && toolDialog === modal) {
        toolDialog = null;
        toolsSummary.focus({ preventScroll: true });
      }
      wasOpen = open;
    }).observe(modal, { attributes: true, attributeFilter: ["class"] });
  }
  window.addEventListener("resize", () => closeTools());
  window.addEventListener("hashchange", () => closeTools());

  const overview = document.querySelector(".practice-overview");
  const hero = overview.querySelector(".hero");
  const statistics = document.createElement("details");
  statistics.className = "practice-stats";
  statistics.innerHTML = '<summary>学习概况<span class="practice-stats-hint">进度与统计</span>' + chevron + '</summary><div class="practice-stats-content"></div>';
  statistics.querySelector(".practice-stats-content").append(overview.querySelector(".hero-progress"), overview.querySelector(".summary-grid"), overview.querySelector(".practice-caption"));
  hero.append(hero.querySelector(".hero-actions"));
  overview.append(statistics);
})();
