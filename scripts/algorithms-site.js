// Site navigation; shared appearance is embedded separately by the sync script.
(() => {
  const style = document.createElement("style");
  style.id = "study-hub-navigation-style";
  style.textContent = `
    #study-hub-home {
      display: inline-flex;
      flex: 0 0 auto;
      align-items: center;
      gap: 4px;
      min-height: 40px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 500;
      line-height: 1.3;
      text-decoration: none;
      white-space: nowrap;
      border-radius: 4px;
    }
    #study-hub-home::after {
      content: "";
      width: 1px;
      height: 24px;
      margin-left: 12px;
      background: var(--line);
    }
    #study-hub-home svg {
      width: 16px;
      height: 16px;
      fill: none;
      stroke: currentColor;
      stroke-width: 1.8;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    #study-hub-home:hover { color: var(--text); }
    #study-hub-home:active { color: var(--accent); }
    .app-header > .header-leading { margin-right: auto; }
    @media (max-width: 1180px) {
      :root { --header-height: 108px; }
      .app-header { padding-top: 40px; }
      #study-hub-home { position: absolute; top: 0; left: 28px; }
      #study-hub-home::after { display: none; }
    }
    @media (max-width: 900px) {
      #study-hub-home { left: 16px; }
    }
    @media (max-width: 540px) {
      :root { --header-height: 104px; }
      #study-hub-home { left: 10px; }
    }
  `;
  document.head.append(style);

  const homeLink = document.createElement("a");
  homeLink.id = "study-hub-home";
  homeLink.href = "../index.html";
  homeLink.setAttribute("aria-label", "返回 Study Hub");
  homeLink.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m14 6-6 6 6 6"></path></svg><span>Study Hub</span>';
  document.getElementById("app-header").prepend(homeLink);
})();
