/* Shared appearance preference, also embedded in the standalone algorithm page. */
(() => {
  const storageKey = 'study-hub:theme';
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  const validTheme = value => value === 'light' || value === 'dark';
  const readPreference = () => {
    try {
      const value = localStorage.getItem(storageKey);
      return validTheme(value) || value === 'system' ? value : null;
    } catch {
      return undefined;
    }
  };
  const writePreference = theme => {
    try {
      localStorage.setItem(storageKey, theme);
      return true;
    } catch {
      return false;
    }
  };

  const stored = readPreference();
  let preference = validTheme(stored) ? stored : null;
  let sessionPreference = false;
  if (stored === null) {
    try {
      // Adopt an existing algorithm preference once, without rewriting learning records.
      const previous = JSON.parse(localStorage.getItem('lc-offline:v1'))?.settings?.theme;
      if (validTheme(previous)) {
        preference = previous;
      }
    } catch { /* Missing, invalid or unavailable storage falls back to the system. */ }
    // Mark migration complete even in automatic mode: algorithm autosaves also contain a theme.
    writePreference(preference || 'system');
  }
  const preferredTheme = () => preference || (systemTheme.matches ? 'dark' : 'light');
  const sun = '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>';
  const moon = '<path d="M20.5 14A8.5 8.5 0 0 1 10 3.5 8.5 8.5 0 1 0 20.5 14Z"/>';

  function refreshControls(root = document) {
    const dark = document.documentElement.dataset.theme === 'dark';
    const label = dark ? '切换浅色模式' : '切换深色模式';
    root.querySelectorAll('[data-theme-toggle]').forEach(button => {
      button.setAttribute('aria-label', label);
      button.title = label;
      button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true">' + (dark ? sun : moon) + '</svg>';
      button.hidden = false;
    });
  }

  function apply(theme) {
    const changed = document.documentElement.dataset.theme !== theme;
    document.documentElement.dataset.theme = theme;
    refreshControls();
    if (changed) window.dispatchEvent(new CustomEvent('study-hub-theme-change', { detail: { theme } }));
  }

  function set(theme, { persist = true } = {}) {
    if (!validTheme(theme)) return;
    if (persist) setPreference(theme);
    else apply(theme);
  }

  function setPreference(value, { persist = true } = {}) {
    if (!validTheme(value) && value !== 'system') return;
    preference = validTheme(value) ? value : null;
    sessionPreference = !persist || !writePreference(value);
    apply(preferredTheme());
  }

  window.StudyHubTheme = Object.freeze({
    get: () => document.documentElement.dataset.theme,
    getPreference: () => preference || 'system',
    set,
    setPreference,
    refreshControls,
  });
  // This script loads before styles so the first paint uses the selected palette.
  apply(preferredTheme());
  document.addEventListener('DOMContentLoaded', () => refreshControls(), { once: true });
  document.addEventListener('click', event => {
    if (event.target.closest('[data-theme-toggle]')) {
      set(window.StudyHubTheme.get() === 'dark' ? 'light' : 'dark');
    }
  });
  systemTheme.addEventListener('change', () => {
    if (!preference) apply(preferredTheme());
  });
  window.addEventListener('storage', event => {
    if (event.key !== storageKey && event.key !== null) return;
    try { if (event.storageArea !== localStorage) return; } catch { return; }
    preference = validTheme(event.newValue) ? event.newValue : null;
    sessionPreference = false;
    apply(preferredTheme());
  });
  // A page restored from the back/forward cache may have missed a storage event.
  window.addEventListener('pageshow', event => {
    if (!event.persisted || sessionPreference) return;
    const stored = readPreference();
    if (stored !== undefined) {
      preference = validTheme(stored) ? stored : null;
      apply(preferredTheme());
    }
  });
})();
