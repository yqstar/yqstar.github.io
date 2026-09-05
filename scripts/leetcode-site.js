// Loaded after PROBLEMS is defined and before the offline app reads its state.
(() => {
  const homeLink = document.createElement("a");
  homeLink.id = "study-hub-home";
  homeLink.href = "index.html";
  homeLink.className = "button secondary";
  homeLink.textContent = "← 返回 Study Hub";
  homeLink.style.cssText = "display:inline-flex;align-items:center;justify-content:center;text-decoration:none";
  document.querySelector(".hero-actions").append(homeLink);

  // Only seed a fresh offline app. Keep both existing offline data and lc_done.
  try {
    if (localStorage.getItem("lc-offline:v1") !== null) return;
    const legacy = JSON.parse(localStorage.getItem("lc_done") || "null");
    if (!Array.isArray(legacy)) return;
    const done = new Set(legacy.filter(Number.isSafeInteger).map(String));
    const records = Object.fromEntries(PROBLEMS
      .filter((problem) => done.has(problem.frontendId))
      .map((problem) => [problem.slug, {
        status: "solved", attempts: 0, note: "", updatedAt: null, passedAt: null,
      }]));
    if (Object.keys(records).length) {
      localStorage.setItem("lc-offline:v1", JSON.stringify({ records, settings: {} }));
    }
  } catch {
    // Corrupt or unavailable legacy storage must never block the new app.
  }
})();
