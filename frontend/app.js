'use strict';

const API = '';

// ── API key (X-API-Key header on every request) ─────────────────────────────
// Admin generates the bootstrap key once (logged to uvicorn stdout on first
// run). Each non-admin key can mint exactly one child key via /keys/generate.
const KEY_STORAGE = 'curemom_api_key';

function getApiKey() {
  return localStorage.getItem(KEY_STORAGE) || '';
}

function setApiKey(k) {
  if (k) localStorage.setItem(KEY_STORAGE, k.trim());
  else localStorage.removeItem(KEY_STORAGE);
}

function ensureApiKey() {
  let k = getApiKey();
  if (!k) {
    k = (prompt('Enter your CureMom API key:\n\n(Ask the admin if you don\'t have one. The key is stored locally in your browser.)') || '').trim();
    if (k) setApiKey(k);
  }
  return k;
}

async function onShareAccessClick() {
  // Mint a child key. Admin can mint unlimited; non-admin gets exactly one.
  // Backend enforces; we surface the result either way.
  const note = (prompt('Optional label for the new key (e.g. "Alice"):') || '').trim();
  try {
    const r = await apiFetch(`${API}/api/v1/keys/generate`, {
      method: 'POST',
      body: JSON.stringify({ note: note || null }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      alert('Could not mint key: ' + _formatErr(err, r));
      return;
    }
    const data = await r.json();
    // Show the new key with copy-to-clipboard. The browser prompt is the
    // simplest cross-browser way to display + let the user copy.
    prompt(
      'Share this key (you will not see it again):\n' +
      (note ? `Label: ${note}\n\n` : '\n'),
      data.key,
    );
    // Refresh the button: a non-admin who just used their one allowance
    // should now see the disabled "limit reached" state.
    _refreshAuthState();
  } catch (err) {
    alert('Could not mint key: ' + (err.message || err));
  }
}

// On page load: if a key is already in localStorage, validate it via
// /keys/me. The Share section stays hidden until we confirm the key is
// good — visitors who haven't entered a key yet shouldn't see UI for
// minting more keys.
async function _refreshAuthState() {
  const section = document.getElementById('share-section');
  const btn = document.getElementById('share-access-btn');
  if (!section || !btn) return;

  const key = getApiKey();
  if (!key) {
    section.style.display = 'none';
    return;
  }
  try {
    const r = await fetch(`${API}/api/v1/keys/me`, {
      headers: { 'X-API-Key': key },
      signal: AbortSignal.timeout(5000),
    });
    if (r.status === 401) {
      // Stored key was rejected — clear it so the next gated call re-prompts.
      setApiKey('');
      section.style.display = 'none';
      return;
    }
    if (!r.ok) return;   // transient error; leave UI alone
    const data = await r.json();
    section.style.display = '';
    if (data.can_mint_more) {
      btn.disabled = false;
      btn.title = data.is_admin
        ? 'Admin: mint a new child key (unlimited)'
        : 'Mint your one child key to share with someone else';
    } else {
      btn.disabled = true;
      btn.title = 'You have already minted your one child key (admin can mint unlimited).';
    }
  } catch {
    // Silent — leave hidden, will retry on next gated call success.
  }
}

// Wrap fetch so every call carries X-API-Key. Same signature as fetch().
async function apiFetch(url, opts = {}) {
  const key = ensureApiKey();
  const headers = new Headers(opts.headers || {});
  headers.set('X-API-Key', key);
  if (opts.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const r = await fetch(url, { ...opts, headers });
  if (r.status === 401) {
    // Bad key — clear it so the next request prompts again.
    setApiKey('');
    alert('API key rejected. Reload to enter a new one.');
  }
  return r;
}

// FastAPI returns errors in three shapes:
//   { detail: "string message" }                — explicit HTTPException
//   { detail: [{type, loc, msg, ...}, ...] }   — Pydantic 422 validation
//   {}                                          — body parse failed
// `${err.detail ?? r.statusText}` ends up rendering "[object Object]" for
// the array case, which is what showed up to the user. This helper coerces
// each shape into a readable single line.
function _formatErr(err, r) {
  const d = err && err.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d.map((e) => {
      const where = Array.isArray(e.loc) ? e.loc.slice(1).join('.') : '';
      return where ? `${where}: ${e.msg}` : (e.msg || JSON.stringify(e));
    }).join('; ');
  }
  if (d && typeof d === 'object') return JSON.stringify(d);
  return r ? `HTTP ${r.status} ${r.statusText}` : 'unknown error';
}

// ── View switching ──────────────────────────────────────────────────────────
function switchMode(mode) {
  ['landing', 'consumer', 'professional'].forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (el) el.classList.toggle('active', v === mode);
  });
  // Cytoscape was likely initialized while the patient view was hidden
  // (canvas had 0×0 dimensions). When the view becomes active, the canvas
  // gains real dimensions but Cytoscape's internal viewport is still
  // stuck at 0×0. Force a resize+fit so existing nodes render correctly.
  if (mode === 'consumer' && typeof KGraph !== 'undefined' && _graphInitialized) {
    setTimeout(() => { try { KGraph.resize(); KGraph.fit(); } catch (_) {} }, 60);
  }
}

// ── Status check ────────────────────────────────────────────────────────────
async function checkStatus() {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  try {
    const r = await fetch(`${API}/api/v1/stats`, { signal: AbortSignal.timeout(4000) });
    if (r.ok) {
      const d = await r.json();
      dot.className = 'status-dot ok';
      const n = d.total_papers ?? d.papers_count ?? '?';
      text.textContent = `${Number(n).toLocaleString()} papers indexed`;
    } else {
      dot.className = 'status-dot warn';
      text.textContent = 'API reachable';
    }
  } catch {
    dot.className = 'status-dot err';
    text.textContent = 'API offline';
  }
}
checkStatus();

// ── Populate provider dropdowns dynamically from /api/v1/llm/status ─────────
async function populateProviderDropdowns() {
  console.log('[providers] fetching /api/v1/llm/status…');
  // No silent fallback — let errors surface so the user knows the dropdown
  // is incomplete because of a real failure, not because we picked defaults.
  const r = await fetch(`${API}/api/v1/llm/status`, { signal: AbortSignal.timeout(5000) });
  if (!r.ok) {
    throw new Error(`/api/v1/llm/status returned HTTP ${r.status}`);
  }
  const status = await r.json();
  console.log('[providers] status:', status);

  const claude = status.providers?.claude || {};
  const openai = status.providers?.openai || {};
  const nim = status.providers?.nim || {};

  for (const id of ['consumer-provider', 'pro-provider']) {
    const sel = document.getElementById(id);
    if (!sel) continue;
    // Strip everything except the static "Extractive" option
    [...sel.options].forEach(o => { if (o.value !== 'extractive') o.remove(); });

    if (claude.available) {
      // Two Claude variants via the backend's `claude/<model>` override:
      // Haiku is the fast/cheap option; Sonnet is the higher-quality option
      // for richer synthesis (used when the user wants better graph extraction).
      const haikuOpt = document.createElement('option');
      haikuOpt.value = 'claude/claude-haiku-4-5-20251001';
      haikuOpt.textContent = 'Claude Haiku 4.5 (fast)';
      sel.appendChild(haikuOpt);

      const sonnetOpt = document.createElement('option');
      sonnetOpt.value = 'claude/claude-sonnet-4-6';
      sonnetOpt.textContent = 'Claude Sonnet 4.6 (best quality)';
      sel.appendChild(sonnetOpt);
    }
    if (openai.available) {
      const opt = document.createElement('option');
      opt.value = 'openai';
      opt.textContent = `OpenAI (${openai.model})`;
      sel.appendChild(opt);
    }
    if (nim.available) {
      // Two NIM variants surfaced explicitly. Llama 3.1 70B is the default
      // because it's non-reasoning — fast (~7 s) on structured-output tasks
      // like graph extraction + merge. MiniMax M2.7 is a reasoning model
      // that emits ~3 K thinking tokens before the answer (~100 s on a
      // 37-node merge); NIM doesn't honor enable_thinking=False on it.
      // MiniMax M2.7 (reasoning) is the default; our system prompt now
      // appends a "do not reason" suffix that empirically cuts its
      // thinking budget enough to be tolerable for graph_extract and
      // dedup. Llama 3.1 70B stays as a non-reasoning escape hatch.
      const minimaxOpt = document.createElement('option');
      minimaxOpt.value = 'nim/minimaxai/minimax-m2.7';
      minimaxOpt.textContent = 'NIM MiniMax M2.7 (reasoning, default)';
      minimaxOpt.dataset.isDefault = '1';
      sel.appendChild(minimaxOpt);

      const llamaOpt = document.createElement('option');
      llamaOpt.value = 'nim/meta/llama-4-maverick-17b-128e-instruct';
      llamaOpt.textContent = 'NIM Llama 4 Maverick (fast, non-reasoning)';
      sel.appendChild(llamaOpt);
    }

    // Default to NIM (free tier) if available, else first non-extractive
    // option, else extractive.
    const def = [...sel.options].find(o => o.dataset.isDefault === '1');
    if (def) sel.value = def.value;

    // Bind mobile bottom-sheet picker. On phones, intercept the select's
    // native picker and show our themed sheet instead.
    _bindProviderSheet(sel);
  }
  console.log('[providers] dropdown populated');
}
populateProviderDropdowns().catch(err => {
  console.error('[providers] populate failed:', err);
});

// ── Mobile bottom-sheet provider picker (Phase 5) ───────────────────────
// On phones (≤700 px), tapping a provider <select> opens a styled
// bottom sheet instead of the native picker. The selected option is
// mirrored back into the hidden <select> so existing code that reads
// `consumer-provider.value` works without any change. Desktop is
// untouched — pointer events bypass the interceptor at >700 px.
function _bindProviderSheet(selectEl) {
  if (!selectEl || selectEl._sheetBound) return;
  selectEl._sheetBound = true;
  // mousedown fires BEFORE the native picker opens, so we preventDefault
  // here and route to our sheet. iOS Safari + Chrome both honor this.
  selectEl.addEventListener('mousedown', (e) => {
    if (!matchMedia('(max-width: 700px)').matches) return;
    e.preventDefault();
    openProviderSheet(selectEl);
  });
  // Keyboard focus still opens the native picker on desktop (good); on
  // phone, also intercept keyboard activation:
  selectEl.addEventListener('keydown', (e) => {
    if (!matchMedia('(max-width: 700px)').matches) return;
    if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') {
      e.preventDefault();
      openProviderSheet(selectEl);
    }
  });
}

function openProviderSheet(selectEl) {
  const sheet = document.getElementById('provider-sheet');
  const list  = document.getElementById('provider-sheet-list');
  if (!sheet || !list) return;
  sheet._target = selectEl;
  list.innerHTML = [...selectEl.options].map((o) => {
    const sel = o.value === selectEl.value ? ' sel' : '';
    return `<li><button class="sheet-opt${sel}" type="button" data-v="${escapeHtml(o.value)}">${escapeHtml(o.textContent)}</button></li>`;
  }).join('');
  list.onclick = (e) => {
    const btn = e.target.closest('.sheet-opt');
    if (!btn) return;
    selectEl.value = btn.dataset.v;
    selectEl.dispatchEvent(new Event('change', { bubbles: true }));
    closeProviderSheet();
  };
  sheet.hidden = false;
  // Two-frame delay so the slide-in transition triggers (initial paint
  // sets transform:translateY(100%); class flip then animates to 0).
  requestAnimationFrame(() => requestAnimationFrame(() => sheet.classList.add('open')));
}

function closeProviderSheet() {
  const sheet = document.getElementById('provider-sheet');
  if (!sheet) return;
  sheet.classList.remove('open');
  setTimeout(() => { sheet.hidden = true; }, 240);
}

window.closeProviderSheet = closeProviderSheet;   // for inline onclick

// ── Settings sheet (mobile-only — opened by ⚙ in chat topbar) ───────────
// Aggregates Plain-language / Provider / Strategy into one sheet so the
// inline chat-controls row can be hidden on phone. Each row either flips
// a switch in place (plain-language) or opens a child sheet (provider,
// strategy) for the actual picker.
function openSettingsSheet() {
  const sheet = document.getElementById('settings-sheet');
  if (!sheet) return;
  // Sync the inline toggle into the sheet on every open.
  const simpleSrc = document.getElementById('consumer-simple');
  const simpleDst = document.getElementById('settings-simple-toggle');
  if (simpleSrc && simpleDst) {
    simpleDst.checked = simpleSrc.checked;
    simpleDst.onchange = () => { simpleSrc.checked = simpleDst.checked; };
  }
  // Show the current provider + strategy labels as hints.
  const pSel = document.getElementById('consumer-provider');
  const sSel = document.getElementById('consumer-strategy');
  const pHint = document.getElementById('settings-provider-current');
  const sHint = document.getElementById('settings-strategy-current');
  if (pSel && pHint) {
    const opt = pSel.options[pSel.selectedIndex];
    pHint.textContent = opt ? opt.textContent : '—';
  }
  if (sSel && sHint) {
    const opt = sSel.options[sSel.selectedIndex];
    sHint.textContent = opt ? opt.textContent : '—';
  }
  sheet.hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() => sheet.classList.add('open')));
}
function closeSettingsSheet() {
  const sheet = document.getElementById('settings-sheet');
  if (!sheet) return;
  sheet.classList.remove('open');
  setTimeout(() => { sheet.hidden = true; }, 240);
}
window.openSettingsSheet = openSettingsSheet;
window.closeSettingsSheet = closeSettingsSheet;

// Strategy bottom sheet — same UX as provider sheet.
function openStrategySheet() {
  const select = document.getElementById('consumer-strategy');
  const sheet  = document.getElementById('strategy-sheet');
  const list   = document.getElementById('strategy-sheet-list');
  if (!select || !sheet || !list) return;
  list.innerHTML = [...select.options].map((o) => {
    const sel = o.value === select.value ? ' sel' : '';
    return `<li><button class="sheet-opt${sel}" type="button" data-v="${escapeHtml(o.value)}" ${o.disabled ? 'disabled' : ''}>${escapeHtml(o.textContent)}</button></li>`;
  }).join('');
  list.onclick = (e) => {
    const btn = e.target.closest('.sheet-opt');
    if (!btn || btn.disabled) return;
    select.value = btn.dataset.v;
    select.dispatchEvent(new Event('change', { bubbles: true }));
    closeStrategySheet();
  };
  sheet.hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() => sheet.classList.add('open')));
}
function closeStrategySheet() {
  const sheet = document.getElementById('strategy-sheet');
  if (!sheet) return;
  sheet.classList.remove('open');
  setTimeout(() => { sheet.hidden = true; }, 240);
}
window.openStrategySheet = openStrategySheet;
window.closeStrategySheet = closeStrategySheet;

// Validate any stored key + reveal the Share section if so.
_refreshAuthState();

// ── Stale-tab detection ────────────────────────────────────────────────────
// Multi-tab scenario: a redeploy bumps app.js's mtime on disk. Tabs opened
// before the redeploy still run the old JS in memory. Poll /api/v1/version
// every 30s; if the mtime differs from the one this tab saw on first load,
// show a non-blocking banner asking the user to reload. Active tab fires
// once on load to seed the baseline.
let _versionBaseline = null;

async function _checkAppVersion() {
  try {
    const r = await fetch(`${API}/api/v1/version`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) return;
    const { app_js_mtime } = await r.json();
    if (!app_js_mtime) return;
    if (_versionBaseline === null) {
      _versionBaseline = app_js_mtime;
      return;
    }
    if (app_js_mtime !== _versionBaseline) _showStaleVersionBanner();
  } catch { /* network blip — ignore */ }
}

function _showStaleVersionBanner() {
  if (document.getElementById('stale-banner')) return;   // already shown
  const div = document.createElement('div');
  div.id = 'stale-banner';
  div.className = 'stale-banner';
  div.innerHTML = `
    <span>A newer version of the app is available.</span>
    <button onclick="location.reload()">Reload</button>
    <button onclick="this.parentElement.remove()" aria-label="Dismiss">×</button>
  `;
  document.body.appendChild(div);
}

_checkAppVersion();
setInterval(_checkAppVersion, 30_000);

// ── Knowledge graph panel ───────────────────────────────────────────────────
// The graph is session-local and grows with each Q&A turn. After every
// answer we POST to /api/v1/graph_extract with the question, the cleaned
// answer text, and the cited chunks; the backend runs NER + LLM JSON-mode
// to emit nodes/edges, and KGraph.merge() folds them into the canvas.
let _graphInitialized = false;
const GRAPH_PREF_KEY = 'curemom.graphPanelOpen';

function setupKnowledgeGraph() {
  // Default the panel to OPEN unless the user has explicitly closed it before.
  const panel = document.getElementById('graph-panel');
  if (!panel) return;
  const savedPref = localStorage.getItem(GRAPH_PREF_KEY);
  // Narrow viewports (iPad portrait) — default the graph panel to closed
  // since the chat area gets cramped. User can still toggle it open.
  const isNarrow = window.matchMedia('(max-width: 900px)').matches;
  const defaultOpen = !isNarrow;
  const startOpen = savedPref === null ? defaultOpen : savedPref === '1';
  if (startOpen) panel.classList.remove('collapsed');
  else panel.classList.add('collapsed');
  document.getElementById('graph-toggle-btn')?.classList.toggle('active', startOpen);

  // Cytoscape init is purely lazy — happens on first _extractGraph call OR
  // first manual toggle. Initializing here at script-load time would put
  // Cytoscape inside a 0×0 canvas (the patient view hasn't been activated
  // yet) and the layout breaks.

  // Wire action buttons
  document.getElementById('graph-clear-btn')?.addEventListener('click', () => {
    if (!_graphInitialized) return;
    KGraph.clear();
    _refreshGraphChrome();
    _hidePopover();
  });
  document.getElementById('graph-export-btn')?.addEventListener('click', () => {
    if (!_graphInitialized) return;
    const payload = KGraph.exportJSON();
    _downloadBlob(
      new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
      `curemom-graph-${_timestamp()}.json`,
    );
  });
  document.getElementById('graph-png-btn')?.addEventListener('click', async () => {
    if (!_graphInitialized) return;
    const blob = KGraph.exportPNG({ scale: 2 });
    if (!blob) return;
    _downloadBlob(blob, `curemom-graph-${_timestamp()}.png`);
  });
  document.getElementById('graph-restore-btn')?.addEventListener('click', () => {
    document.getElementById('graph-restore-file')?.click();
  });
  document.getElementById('graph-restore-file')?.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';  // allow re-selecting the same file later
    if (!file) return;
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      if (!payload || !Array.isArray(payload.nodes) || !Array.isArray(payload.edges)) {
        alert('That file doesn\'t look like a CureMom graph export.');
        return;
      }
      ensureGraphInit();
      const ok = KGraph.restoreFromPayload(payload);
      if (ok) {
        _refreshGraphChrome();
      } else {
        alert('Restore failed — payload was rejected.');
      }
    } catch (err) {
      console.error('[KGraph] restore failed:', err);
      alert('Could not restore graph: ' + (err.message || err));
    }
  });
  document.getElementById('graph-zoom-in-btn')?.addEventListener('click', () => {
    ensureGraphInit();
    KGraph.zoomBy(1.25);
  });
  document.getElementById('graph-zoom-out-btn')?.addEventListener('click', () => {
    ensureGraphInit();
    KGraph.zoomBy(0.8);
  });
  document.getElementById('graph-fit-btn')?.addEventListener('click', () => {
    ensureGraphInit();
    KGraph.fit();
  });
  document.getElementById('graph-merge-btn')?.addEventListener('click', _onMergeClick);

  const searchInput = document.getElementById('graph-search');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      ensureGraphInit();
      KGraph.searchNodes(e.target.value);
    });
    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        e.target.value = '';
        KGraph.searchNodes('');
        e.target.blur();
      }
    });
  }

  // When the panel is shown after being hidden, the canvas needs a resize.
  // Also resize on window resize so labels don't get clipped.
  window.addEventListener('resize', () => {
    if (_graphInitialized && _isGraphPanelOpen()) KGraph.resize();
  });
}

function ensureGraphInit() {
  if (_graphInitialized) return;
  if (typeof KGraph === 'undefined') {
    console.warn('[KGraph] init skipped: KGraph module not loaded yet');
    return;
  }
  const canvas = document.getElementById('graph-canvas');
  if (!canvas) {
    console.warn('[KGraph] init skipped: #graph-canvas element not in DOM');
    return;
  }
  const rect = canvas.getBoundingClientRect();
  console.log('[KGraph] init — canvas size at init:', rect.width, '×', rect.height);
  const cy = KGraph.init(canvas);
  if (!cy) {
    console.error('[KGraph] init failed — KGraph.init returned null. Is cytoscape loaded?');
    return;
  }
  KGraph.onNodeClick(_onGraphNodeClick);
  _graphInitialized = true;
  _refreshGraphChrome();
  // Force a resize after a microtask so any layout-pending dimensions resolve.
  setTimeout(() => { try { KGraph.resize(); } catch (_) {} }, 30);
}

function toggleGraphPanel() {
  const panel = document.getElementById('graph-panel');
  if (!panel) return;
  const isCollapsed = panel.classList.toggle('collapsed');
  document.getElementById('graph-toggle-btn')?.classList.toggle('active', !isCollapsed);
  localStorage.setItem(GRAPH_PREF_KEY, isCollapsed ? '0' : '1');
  if (!isCollapsed) {
    ensureGraphInit();
    // Cytoscape needs a resize/fit after the container becomes visible.
    setTimeout(() => {
      if (window.cytoscape && document.getElementById('graph-canvas')?._cyreg?.cy) {
        document.getElementById('graph-canvas')._cyreg.cy.resize();
        document.getElementById('graph-canvas')._cyreg.cy.fit(undefined, 30);
      }
    }, 50);
  } else {
    _hidePopover();
  }
}

function _isGraphPanelOpen() {
  const panel = document.getElementById('graph-panel');
  return panel && !panel.classList.contains('collapsed');
}

// ── Mobile tab swap (Phase 2 + 3) ───────────────────────────────────────
// On phones (≤900 px CSS breakpoint), the .app-shell carries a
// data-mobile-tab attribute that flips which panel is visible. The
// bottom-nav buttons call this. Desktop ignores it (CSS only reacts
// at ≤900 px).
function setMobileTab(tab) {
  const shell = document.querySelector('#view-consumer .app-shell');
  if (!shell) return;
  const next = tab === 'graph' ? 'graph' : 'chat';
  if (shell.dataset.mobileTab === next) return;

  const swap = () => {
    shell.dataset.mobileTab = next;
    document.querySelectorAll('#view-consumer .bn-btn').forEach((b) => {
      b.setAttribute('aria-selected', b.dataset.tab === next ? 'true' : 'false');
    });
    if (next === 'graph') {
      ensureGraphInit();
      // Cytoscape needs a resize + fit after its container becomes visible.
      setTimeout(() => {
        try { KGraph.resize(); KGraph.fit(); } catch (_) { /* not yet inited */ }
      }, 60);
    } else {
      _hidePopover?.();
    }
    // Clear the unread-count badge whenever the graph tab is visited.
    if (next === 'graph') {
      const badge = document.getElementById('bn-graph-badge');
      if (badge) { badge.hidden = true; badge.textContent = '0'; }
    }
  };

  // Smoother transition via View Transitions API where supported.
  if (document.startViewTransition && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.startViewTransition(swap);
  } else {
    swap();
  }
}

// Bump the bottom-nav badge when new graph content arrives while the
// user is on the chat tab. Called from _extractGraph after a successful
// merge. No-op on desktop (where both panels are always visible).
function _bumpGraphBadge(added) {
  if (!added) return;
  const shell = document.querySelector('#view-consumer .app-shell');
  const badge = document.getElementById('bn-graph-badge');
  if (!shell || !badge) return;
  if (shell.dataset.mobileTab === 'graph') return;   // already viewing
  if (!matchMedia('(max-width: 900px)').matches) return;
  const cur = parseInt(badge.textContent || '0', 10) || 0;
  badge.textContent = String(cur + added);
  badge.hidden = false;
}

window.setMobileTab = setMobileTab;   // expose for inline onclick

function _refreshGraphChrome(overrideStatus) {
  if (!_graphInitialized) return;
  const { nodes, edges } = KGraph.size();
  const empty = document.getElementById('graph-empty');
  const status = document.getElementById('graph-status');
  if (empty) empty.classList.toggle('hidden', nodes > 0);
  if (status) {
    if (overrideStatus) {
      status.textContent = overrideStatus;
    } else {
      status.textContent = nodes > 0 ? `${nodes} node${nodes !== 1 ? 's' : ''} · ${edges} edge${edges !== 1 ? 's' : ''}` : '';
    }
  }
}

async function _onMergeClick() {
  // Every branch surfaces a chrome message — earlier versions swallowed
  // skips and failures into console.error only, which looked like "the
  // button doesn't work" from the user's side.
  if (!_graphInitialized) {
    console.log('[KGraph] merge skipped — graph not initialized');
    _refreshGraphChrome('Merge skipped — open the graph panel first');
    setTimeout(() => _refreshGraphChrome(), 4000);
    return;
  }
  const { nodes } = KGraph.size();
  if (nodes < 2) {
    console.log('[KGraph] merge skipped — need at least 2 nodes');
    _refreshGraphChrome('Merge skipped — need at least 2 nodes');
    setTimeout(() => _refreshGraphChrome(), 4000);
    return;
  }
  const labels = KGraph.exportJSON().nodes.map(n => n.label);
  const provider = document.getElementById('consumer-provider')?.value || null;
  const btn = document.getElementById('graph-merge-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Merging…'; }
  _refreshGraphChrome(`Merging ${labels.length} labels via ${provider || 'default'}…`);
  console.log('[KGraph] POST /api/v1/graph_dedup with', labels.length, 'labels, provider:', provider);
  console.log('[KGraph] submitted labels:', labels);
  try {
    const payload = await _runGraphJob({
      startUrl: `${API}/api/v1/graph_dedup`,
      jobUrlPrefix: `${API}/api/v1/graph_dedup/job`,
      startBody: { labels, llm_provider: provider },
      pollMs: 750,  // tighter cadence — dedup output is short, don't waste a 1.5 s tail
      onProgress: (s) => {
        _refreshGraphChrome(`Merging via ${provider || 'default'}… ${s.elapsed_s}s`);
        if (btn) btn.textContent = `Merging… ${s.elapsed_s}s`;
      },
    });
    if (!payload) {
      _refreshGraphChrome('Merge returned no payload');
      setTimeout(() => _refreshGraphChrome(), 6000);
      return;
    }
    const groups = payload.groups || [];
    console.log('[KGraph] received', groups.length, 'merge groups:', groups);
    const before = KGraph.size().nodes;
    const result = KGraph.applyMergeGroups(groups);
    // Re-check duplicates AFTER the merge — typically clears the
    // hi-vis state, but a partial merge may leave residual dupes.
    _detectAndFlagDuplicates();
    const after = KGraph.size().nodes;
    const status = result.groupsApplied > 0
      ? `Merged ${result.groupsApplied} group${result.groupsApplied !== 1 ? 's' : ''} (${before} → ${after} nodes)`
      : `No duplicates found by ${provider || 'default'} (LLM saw ${labels.length} labels)`;
    _refreshGraphChrome(status);
    setTimeout(() => _refreshGraphChrome(), 6000);
  } catch (err) {
    console.error('[KGraph] dedup failed:', err);
    _refreshGraphChrome(`Merge failed: ${(err && err.message) || err}`);
    setTimeout(() => _refreshGraphChrome(), 8000);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Merge'; }
  }
}

function _showGraphSpinner(show) {
  const sp = document.getElementById('graph-spinner');
  if (sp) sp.hidden = !show;
}

// Skeleton shimmer over the canvas while the LLM is generating relations.
// Rendered into .graph-canvas-wrap once; hidden via [hidden] thereafter.
function _showGraphSkeleton(show) {
  const wrap = document.querySelector('.graph-canvas-wrap');
  if (!wrap) return;
  let el = document.getElementById('graph-skeleton');
  if (!el && show) {
    el = document.createElement('div');
    el.id = 'graph-skeleton';
    el.className = 'graph-skeleton';
    el.innerHTML =
      Array.from({ length: 6 }, (_, i) => `<span class="skel-node n${i}"></span>`).join('') +
      '<svg class="skel-edges" viewBox="0 0 400 300" preserveAspectRatio="none">' +
        '<path d="M 90 70 Q 200 130 240 130 T 310 80"/>' +
        '<path d="M 240 130 Q 200 200 130 200"/>' +
        '<path d="M 130 200 Q 220 240 290 220"/>' +
        '<path d="M 90 70 Q 160 60 220 50"/>' +
      '</svg>';
    wrap.appendChild(el);
  }
  if (el) el.hidden = !show;
}

// Clean the assistant response into prose suitable for graph extraction.
// Same idea as pushAssistantToHistory but kept local so we can call it
// before pushing to history.
function _cleanForGraph(text) {
  let t = text || '';
  t = t.replace(/(?:\n|^)\s*(?:\*+|#+)?\s*(?:you\s+might\s+also\s+want\s+to\s+know|follow[-\s]?up\s+questions|suggested\s+questions|related\s+questions)\s*:?\s*\*?\*?[\s\S]*$/i, '');
  t = t.replace(/this is for information only[^\n]*/gi, '');
  t = t.replace(/\[\d+\]/g, '');
  t = t.replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  return t;
}

// Monotonic token incremented on every _extractGraph call. A second Q&A
// fired before the previous graph extraction returns causes the previous
// call's awaited result to be discarded — otherwise the older Q1 result
// can land AFTER Q2's, re-shuffle the canvas, and flicker the spinner.
let _graphExtractToken = 0;

async function _extractGraph(query, response, citations) {
  const token = ++_graphExtractToken;
  console.log('[KGraph] _extractGraph called — query:', query.slice(0, 60), '| panel open:', _isGraphPanelOpen(), '| token:', token);
  // Always run — even if the panel is closed, the data accumulates so opening
  // the panel later shows what was collected. Spinner is panel-only.
  ensureGraphInit();
  const cleanAnswer = _cleanForGraph(response);
  if (!cleanAnswer || cleanAnswer.length < 10) {
    console.warn('[KGraph] skipped — answer too short after cleaning');
    return;
  }
  const chunks = (citations || [])
    .map(c => ({ id: c.chunk_id, text: c.chunk?.text || '' }))
    .filter(c => Number.isFinite(c.id) && c.text);
  if (!chunks.length) {
    console.warn('[KGraph] skipped — no usable chunks (need {chunk_id, chunk.text})');
    return;
  }
  // Reuse the same provider the user picked for QA so the answer and the
  // graph come from the same brain (only when it's an Ollama model — for
  // claude/openai/extractive the backend falls back to its env default).
  const provider = document.getElementById('consumer-provider')?.value || null;
  console.log('[KGraph] POST /api/v1/graph_extract with', chunks.length, 'chunks, provider:', provider);

  if (_isGraphPanelOpen()) _showGraphSpinner(true);
  _showGraphSkeleton(true);
  // Run the LLM call as a background job, poll until done. Each HTTP
  // round-trip is short and survives any tunnel cap. The job itself can
  // run as long as the LLM needs (up to 180 s wall-clock).
  let payload = null;
  let lastErr = null;
  try {
    try {
      payload = await _runGraphJob({
        startUrl: `${API}/api/v1/graph_extract`,
        jobUrlPrefix: `${API}/api/v1/graph_extract/job`,
        startBody: { query, answer: cleanAnswer, chunks, llm_provider: provider },
        onProgress: (s) => {
          // Only the most-recent extract owns the chrome — a stale tick
          // from an older call must not overwrite the newer one's state.
          if (token !== _graphExtractToken) return;
          _refreshGraphChrome(`Extracting graph via ${provider || 'default'}… ${s.elapsed_s}s`);
        },
      });
    } catch (err) {
      lastErr = err;
      console.error('[KGraph] graph_extract job failed (token', token, '):', err?.message);
    }
    // Stale result: another _extractGraph started while we were polling.
    // Discard ours — the newer one owns the spinner + canvas now.
    if (token !== _graphExtractToken) {
      console.log('[KGraph] discarding stale graph_extract result — token', token, '!=', _graphExtractToken);
      return;
    }
    if (!payload) {
      console.error('[KGraph] /graph_extract fetch failed after retries:', lastErr);
      return;
    }
    console.log('[KGraph] received payload:', (payload.nodes || []).length, 'nodes,', (payload.edges || []).length, 'edges');
    if (payload.error) {
      console.error('[KGraph] backend reported error:', payload.error);
    }
    if (!payload.nodes?.length && !payload.edges?.length) {
      if (payload.error) {
        console.warn('[KGraph] empty payload due to error above.');
      } else {
        console.warn('[KGraph] empty payload — LLM produced no grounded entities/relations for this answer');
      }
      return;
    }
    const result = KGraph.merge(payload);
    console.log('[KGraph] merged —', result.addedNodes.length, 'new nodes,', result.addedEdges.length, 'new edges');
    _bumpGraphBadge(result.addedNodes.length);   // bottom-nav badge on mobile when user is on chat tab
    _detectAndFlagDuplicates();                  // nudge Merge button when likely dupes exist
    // Partial payload: graph built from a truncated LLM input or output.
    // Surface the specific reason in the graph chrome (input vs output is
    // actionable — input truncation means shorten the question; output
    // truncation means raise max_tokens).
    if (payload.error && /^truncated/i.test(payload.error)) {
      const reason = payload.error.split(';')[0].trim();
      _refreshGraphChrome(`Graph partial — ${reason}`);
      setTimeout(() => {
        if (token === _graphExtractToken) _refreshGraphChrome();
      }, 8000);
    } else {
      _refreshGraphChrome();
    }
    // If the panel is collapsed but we have new content, gently nudge it open
    // so the user sees the result.
    if (!_isGraphPanelOpen() && (result.addedNodes.length || result.addedEdges.length)) {
      console.log('[KGraph] auto-opening panel — first content arrived');
      toggleGraphPanel();
    }
  } finally {
    // Only the most-recent call owns the spinner — a stale one finishing
    // late must not turn off the spinner the newer call just turned on.
    if (token === _graphExtractToken) {
      _showGraphSpinner(false);
      _showGraphSkeleton(false);
    }
  }
}

// ── Node-click popover ──────────────────────────────────────────────────────
// `nodePayload` is null when the user clicks the canvas background.
function _onGraphNodeClick(nodePayload) {
  if (!nodePayload) { _hidePopover(); return; }
  const wrap = document.querySelector('.graph-canvas-wrap');
  const pop  = document.getElementById('graph-popover');
  if (!wrap || !pop) return;

  // Build popover content
  const cites = (nodePayload.citations || []).slice().sort((a, b) => a - b);
  const citePills = cites.length
    ? `<div class="popover-citations">${cites.map(c => {
        // Try to map chunk_id back to a citation index in the most recent
        // assistant bubble's pill list. If found, render the index for clarity;
        // otherwise show the raw chunk id as a fallback.
        const idx = _lookupCitationIndexByChunkId(c);
        return `<span class="popover-cite-pill" data-cid="${c}">${idx ? `[${idx}]` : `c${c}`}</span>`;
      }).join('')}</div>`
    : `<div class="popover-empty-cites">No citations from the most recent answer mention this entity yet.</div>`;

  // Type-specific quick actions. Disease → Cure; Drug/Gene → +/−
  // (search for things that promote or inhibit the substance/protein).
  // Other types just get Ask + Remove like before.
  let typeActions = '';
  if (nodePayload.type === 'DISEASE') {
    typeActions = `<button class="popover-action-btn popover-cure-btn" title="Search for medicines that cure or treat this disease">Cure</button>`;
  } else if (nodePayload.type === 'DRUG' || nodePayload.type === 'GENE') {
    typeActions = `
      <button class="popover-action-btn popover-promote-btn" title="Search for medicines that promote or increase this">+</button>
      <button class="popover-action-btn popover-suppress-btn" title="Search for medicines that decrease or inhibit this">−</button>
    `;
  }

  pop.innerHTML = `
    <button class="popover-close-x" aria-label="Close">×</button>
    <div class="popover-header">
      <div class="popover-label">${escapeHtml(nodePayload.label)}</div>
      <div class="popover-type-chip">${escapeHtml(_humanizeType(nodePayload.type))}</div>
    </div>
    ${citePills}
    <div class="popover-actions">
      <button class="popover-ask-btn">Ask about this</button>
      ${typeActions}
      <button class="popover-remove-btn" title="Remove this node from the graph">Remove</button>
    </div>
  `;

  // Position near the node, clamped to the canvas wrap
  const wrapRect = wrap.getBoundingClientRect();
  const pos = nodePayload.renderedPosition || { x: wrapRect.width / 2, y: wrapRect.height / 2 };
  pop.style.left = Math.max(10, Math.min(wrapRect.width - 280, pos.x + 18)) + 'px';
  pop.style.top  = Math.max(10, Math.min(wrapRect.height - 140, pos.y + 12)) + 'px';
  pop.hidden = false;

  pop.querySelector('.popover-close-x')?.addEventListener('click', _hidePopover);
  pop.querySelector('.popover-ask-btn')?.addEventListener('click', () => {
    _prefillChatFromGraph(`Tell me more about ${nodePayload.label} in this context.`);
  });
  pop.querySelector('.popover-cure-btn')?.addEventListener('click', () => {
    _prefillChatFromGraph(`What medicines or treatments are known to cure or treat ${nodePayload.label}? Include both first-line therapies and emerging interventions.`);
  });
  pop.querySelector('.popover-promote-btn')?.addEventListener('click', () => {
    _prefillChatFromGraph(`What medicines, supplements, or interventions are known to promote, increase, or upregulate ${nodePayload.label}?`);
  });
  pop.querySelector('.popover-suppress-btn')?.addEventListener('click', () => {
    _prefillChatFromGraph(`What medicines, drugs, or interventions are known to inhibit, decrease, or suppress ${nodePayload.label}?`);
  });
  pop.querySelector('.popover-remove-btn')?.addEventListener('click', () => {
    if (!_graphInitialized) return;
    KGraph.removeNode(nodePayload.id);
    _refreshGraphChrome();
    _hidePopover();
  });
}

function _prefillChatFromGraph(text) {
  const ta = document.getElementById('consumer-input');
  if (!ta) return;
  ta.value = text;
  autoResize(ta);
  ta.focus();
  _hidePopover();
}

function _hidePopover() {
  const pop = document.getElementById('graph-popover');
  if (pop) pop.hidden = true;
}

function _humanizeType(type) {
  switch (type) {
    case 'CHEMICAL': return 'Drug / chemical';
    case 'DISEASE': return 'Disease';
    case 'GENE_OR_GENE_PRODUCT': return 'Gene / protein';
    case 'ANATOMY': return 'Anatomy';
    case 'SYMPTOM': return 'Symptom';
    case 'PROCEDURE': return 'Procedure';
    case 'CELL_TYPE': return 'Cell type';
    case 'ORGANISM': return 'Organism';
    default: return 'Entity';
  }
}

// Map a chunk_id to its 1-based citation index in the latest AI bubble, if
// such a mapping exists. The bubble already encodes chunk_id via its data
// attributes when rendered (we do this below); falls back to null.
function _lookupCitationIndexByChunkId(chunkId) {
  const bubbles = document.querySelectorAll('.msg-ai .citation-pill[data-chunk-id]');
  for (const el of bubbles) {
    if (parseInt(el.dataset.chunkId, 10) === chunkId) {
      const num = el.querySelector('.cite-num');
      if (num) return parseInt(num.textContent, 10);
    }
  }
  return null;
}

// Initialize once DOM is ready (this script is loaded at end of body, so
// the elements are already there).
setupKnowledgeGraph();

// ── Consumer chat ────────────────────────────────────────────────────────────
// Rolling chat history sent with each request so the LLM has multi-turn
// context. Past assistant turns are stored WITHOUT [N] markers and
// disclaimer (frontend strips before pushing) so the LLM sees clean prose
// and isn't confused by stale citation indices.
const chatHistory = [];           // [{role:'user'|'assistant', content:str}]
const MAX_HISTORY_TURNS = 6;      // keep last 6 user/assistant pairs

function fillExample(btn) {
  const ta = document.getElementById('consumer-input');
  ta.value = btn.textContent.trim();
  autoResize(ta);
  ta.focus();
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendConsumerMessage(); }
}

async function sendConsumerMessage() {
  const ta = document.getElementById('consumer-input');
  const query = ta.value.trim();
  if (!query) return;

  appendUserBubble(query);
  chatHistory.push({ role: 'user', content: query });
  ta.value = '';
  autoResize(ta);

  const btn = document.getElementById('consumer-send');
  btn.disabled = true;
  const typing = appendTypingBubble();

  const provider = document.getElementById('consumer-provider').value;
  const simple   = document.getElementById('consumer-simple').checked;
  const strategy = document.getElementById('consumer-strategy')?.value || 'full';
  const reqBody = JSON.stringify({
    query,
    options: { top_k: 12, retrieval_strategy: strategy, llm_provider: provider, plain_language: simple },
    history: chatHistory.slice(-MAX_HISTORY_TURNS * 2),  // last N user+assistant pairs
  });

  // Start a background QA job and poll for progress. Each HTTP round-trip
  // stays well under any tunnel cap; the LLM can take as long as it needs.
  try {
    const start = await apiFetch(`${API}/api/v1/query/async`, {
      method: 'POST',
      body: reqBody,
      signal: AbortSignal.timeout(15000),
    });
    if (!start.ok) {
      removeTypingBubble(typing);
      const err = await start.json().catch(() => ({}));
      appendAIBubble(`Sorry, I couldn't get a response. ${_formatErr(err, start)}`, []);
      btn.disabled = false;
      return;
    }
    const { job_id } = await start.json();

    // Poll the job, updating the typing-bubble stage on each tick. First
    // poll fires almost immediately so quick answers (cached embedder +
    // small corpus) don't sit on the 1500 ms idle that follows. Each poll
    // fetch has its own 8 s timeout so a dropped tunnel can't stall the
    // overall deadline check.
    const maxWaitMs = 600000;  // 10 min — matches graph deadline + LLM timeout
    const pollMs = 1500;
    const deadline = Date.now() + maxWaitMs;
    let firstPoll = true;
    let lastStage = null;
    let finalPayload = null;
    let finalErr = null;
    while (Date.now() < deadline) {
      await new Promise(res => setTimeout(res, firstPoll ? 150 : pollMs));
      firstPoll = false;
      let pr;
      try {
        pr = await apiFetch(`${API}/api/v1/query/job/${encodeURIComponent(job_id)}`, {
          signal: AbortSignal.timeout(8000),
        });
      } catch {
        // Transient network blip or per-fetch timeout — keep polling.
        continue;
      }
      if (!pr.ok) {
        if (pr.status === 404) { finalErr = new Error('job expired or unknown'); break; }
        continue;  // transient 5xx
      }
      const job = await pr.json().catch(() => null);
      if (!job) continue;
      if (job.stage && job.stage !== lastStage) {
        lastStage = job.stage;
        _setTypingStage(typing, { stage: job.stage, model: job.model });
      }
      if (job.status === 'done') { finalPayload = job.payload; break; }
      if (job.status === 'error') { finalErr = new Error(job.error || 'pipeline error'); break; }
    }

    removeTypingBubble(typing);
    if (finalErr) {
      appendAIBubble(`Sorry, I couldn't get a response. ${finalErr.message}`, []);
      console.error('[query] job failed:', finalErr);
      btn.disabled = false;
      return;
    }
    if (!finalPayload) {
      appendSystemErrorBubble('Timed out waiting for the answer.');
      btn.disabled = false;
      return;
    }
    const response = finalPayload.response ?? 'No response returned.';
    appendAIBubble(response, finalPayload.citations ?? []);
    _refreshAuthState();
    pushAssistantToHistory(response);
    _extractGraph(query, response, finalPayload.citations ?? []);
  } catch (err) {
    removeTypingBubble(typing);
    console.error('[query] failed:', err);
    appendSystemErrorBubble('Could not reach the API. ' + (err?.message || err));
  } finally {
    btn.disabled = false;
  }
}

function appendUserBubble(text) {
  const msgs = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'msg msg-user';
  d.innerHTML = `<div class="msg-content">${escapeHtml(text)}</div>`;
  msgs.appendChild(d);
  scrollChat();
}

// Stage labels are driven by real SSE events from /api/v1/query/stream
// (server emits {stage:"analyzing"|"embedding"|"loading_graph"|"retrieving"|
// "drug_lookup"|"synthesizing"|"verifying"|"complete"}) so the user sees
// where the backend actually is, not where the wall clock guesses it is.
const _STAGE_LABELS = {
  analyzing:     'Analyzing your question…',
  embedding:     'Embedding query for dense search…',
  loading_graph: 'Loading the entity graph (one-time)…',
  retrieving:    'Searching the literature…',
  drug_lookup:   'Looking up drug references…',
  synthesizing:  'Composing the answer…',
  verifying:     'Verifying citations…',
  fallback:      'Connection unstable — retrying without streaming…',
};

function appendTypingBubble() {
  const msgs = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'msg msg-ai';
  d.innerHTML = `
    <div class="msg-avatar">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 2C5.24 2 3 4.24 3 7C3 9.76 5.24 12 8 12C10.76 12 13 9.76 13 7" stroke="white" stroke-width="1.5" stroke-linecap="round"/><path d="M7 5H9M8 3V9" stroke="white" stroke-width="1.5" stroke-linecap="round"/></svg>
    </div>
    <div class="msg-content typing-bubble">
      <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
      <div class="typing-stage" data-typing-stage>Connecting…</div>
    </div>`;
  msgs.appendChild(d);
  scrollChat();
  return d;
}

function _setTypingStage(bubbleEl, evt) {
  const stageEl = bubbleEl?.querySelector('[data-typing-stage]');
  if (!stageEl) return;
  const base = _STAGE_LABELS[evt.stage] || evt.stage;
  // Decorate "synthesizing" with the model so users know which provider is working.
  const label = (evt.stage === 'synthesizing' && evt.model)
    ? `Composing the answer with ${evt.model}…`
    : base;
  stageEl.textContent = label;
}

function removeTypingBubble(el) { el?.remove(); }

function appendSystemErrorBubble(text) {
  // Distinct styling from appendAIBubble — no AI avatar, amber color, an
  // explicit "System error" label. Important: users were mistaking the
  // generic "Could not reach the API" fallback for the LLM speaking.
  const msgs = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'msg msg-system';
  d.innerHTML = `
    <div class="msg-content system-error">
      <div class="system-error-label">⚠ System error</div>
      <div class="system-error-text"></div>
      <div class="system-error-hint">Not from the AI — the browser couldn't talk to the server. Check your network, the API key, or whether the URL has rotated.</div>
    </div>`;
  d.querySelector('.system-error-text').textContent = text;
  msgs.appendChild(d);
  scrollChat();
}

function appendAIBubble(text, citations) {
  const msgs = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'msg msg-ai';

  // Pull off the follow-up section before linkifying. Accepts:
  //   "**You might also want to know:**" / plain "You might also want to know:"
  //   "## Follow-up questions" / "Suggested questions:" / "Related questions:"
  //   marker followed by either newline-bullets OR inline " - " separators
  // No boundary anchor — the LLM sometimes places the marker right after a
  // citation marker ([4]) instead of sentence-ending punctuation. We rely on
  // the marker phrase itself being distinctive enough.
  let mainText = text;
  let followups = [];
  const markerRe = new RegExp(
    String.raw`(?:\*{0,2}|#+)\s*` +
    String.raw`(?:you\s+might\s+also\s+want\s+to\s+know|` +
    String.raw`follow[-\s]?up\s+questions|` +
    String.raw`suggested\s+questions|` +
    String.raw`related\s+questions|` +
    String.raw`questions?\s+you\s+might\s+(?:ask|consider|wonder))` +
    String.raw`\s*:?\s*\*{0,2}\s*`,
    'gi'
  );
  // Take the FIRST match — strip everything after it from the bubble
  // (handles cases where the model duplicates its answer + follow-ups).
  const m = markerRe.exec(text);
  if (m) {
    mainText = text.slice(0, m.index).trimEnd();
    let tail = text.slice(m.index + m[0].length).trim();

    // Drop a duplicate Answer/Conclusion/etc. block if present
    tail = tail.split(
      /\n*\s*\*{0,2}\s*(?:answer|conclusion|in\s+summary|to\s+summari[sz]e)\s*:?\s*\*{0,2}\s*\n/i
    )[0];

    // Try newline-separated bullets first
    let items = tail.split(/\n+/).map(l => l.trim()).filter(Boolean);
    // Fall back to inline " - " / " • " separators if everything's on one line
    if (items.length <= 1 && /\s[-•*]\s/.test(tail)) {
      items = tail.split(/\s+[-•*]\s+/).filter(Boolean);
    }

    followups = items
      .map(l => l.replace(/^\s*[-•→*\d.]+\s*/, '').replace(/\*+/g, '').trim())
      .filter(l => l.length > 5 && l.length < 250)
      // Dedupe in case the model repeated the same questions
      .filter((q, i, arr) => arr.indexOf(q) === i);
  }

  const linked = mainText.replace(/\[(\d+)\]/g, (_, n) => {
    const c = citations[parseInt(n, 10) - 1];
    return c
      ? `<button class="citation-ref" onclick="openModal(${JSON.stringify(c).replace(/"/g, '&quot;')})">[${n}]</button>`
      : `[${n}]`;
  });

  let citeHtml = '';
  if (citations.length) {
    const pills = citations.slice(0, 5).map((c, i) => `
      <button class="citation-pill" data-chunk-id="${c.chunk_id}" onclick="openModal(${JSON.stringify(c).replace(/"/g, '&quot;')})">
        <div class="cite-num">${i + 1}</div>
        <div class="cite-meta">
          <strong>${escapeHtml(c.title ?? 'Unknown title')}</strong><br>
          ${escapeHtml(c.authors ?? '')} ${c.year ? `(${c.year})` : ''} · ${escapeHtml(c.journal ?? '')}
        </div>
      </button>`).join('');
    citeHtml = `
      <button class="citations-toggle" onclick="this.nextElementSibling.hidden=!this.nextElementSibling.hidden">
        📄 ${citations.length} source${citations.length !== 1 ? 's' : ''}
      </button>
      <div class="citations-block" hidden>${pills}</div>`;
  }

  let followupHtml = '';
  if (followups.length) {
    followupHtml = `
      <div class="followups">
        <div class="followups-label">You might also want to know:</div>
        ${followups.map(q => {
          const safe = escapeHtml(q);
          return `<button class="followup-chip" onclick="askFollowup(${JSON.stringify(q).replace(/"/g, '&quot;')})">${safe}</button>`;
        }).join('')}
      </div>`;
  }

  d.innerHTML = `
    <div class="msg-avatar">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 2C5.24 2 3 4.24 3 7C3 9.76 5.24 12 8 12C10.76 12 13 9.76 13 7" stroke="white" stroke-width="1.5" stroke-linecap="round"/><path d="M7 5H9M8 3V9" stroke="white" stroke-width="1.5" stroke-linecap="round"/></svg>
    </div>
    <div class="msg-content"><div data-stream-target></div>${citeHtml}${followupHtml}</div>`;
  msgs.appendChild(d);
  _streamHtmlInto(d.querySelector('[data-stream-target]'), linked, 1500);
  scrollChat();
}

// Sentence-stream `html` into `host`. Walks the parsed DOM, wraps text
// runs in `<span class="stream-piece">`, and reveals them on a stagger
// scaled to fit `durationMs`. Respects prefers-reduced-motion (renders
// the full content atomically). The HTML contains citation buttons
// (.citation-ref) which we treat as atomic non-text nodes.
function _streamHtmlInto(host, html, durationMs) {
  if (!host) return;
  // Reduced-motion: behave like the original atomic append.
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
    host.innerHTML = html;
    return;
  }

  const tmp = document.createElement('div');
  tmp.innerHTML = html;

  // Walk all text nodes; split each by sentence-ish boundaries; replace
  // the text node with a sequence of <span class="stream-piece"> wrappers.
  const pieces = [];
  const _split = (text) => {
    // Split into chunks on sentence punctuation (.?!) followed by space
    // or end. Keep the trailing whitespace attached to the previous chunk.
    const out = [];
    let buf = '';
    for (let i = 0; i < text.length; i++) {
      buf += text[i];
      if (/[.?!]/.test(text[i]) && (text[i + 1] === ' ' || i === text.length - 1)) {
        // Include the trailing space if present.
        if (text[i + 1] === ' ') { buf += ' '; i++; }
        out.push(buf); buf = '';
      }
    }
    if (buf) out.push(buf);
    return out;
  };

  const walker = document.createTreeWalker(tmp, NodeFilter.SHOW_TEXT);
  const targets = [];
  while (walker.nextNode()) targets.push(walker.currentNode);
  for (const node of targets) {
    const text = node.nodeValue;
    if (!text || !text.trim()) continue;
    const chunks = _split(text);
    if (chunks.length === 0) continue;
    const frag = document.createDocumentFragment();
    for (const ch of chunks) {
      const span = document.createElement('span');
      span.className = 'stream-piece';
      span.textContent = ch;
      frag.appendChild(span);
      pieces.push(span);
    }
    node.parentNode.replaceChild(frag, node);
  }

  host.replaceChildren(...tmp.childNodes);

  if (pieces.length === 0) return;
  const per = Math.max(30, Math.floor(durationMs / pieces.length));
  pieces.forEach((el, i) => {
    setTimeout(() => {
      el.classList.add('shown');
      // Re-scroll periodically while text grows.
      if (i % 4 === 0 || i === pieces.length - 1) scrollChat();
    }, per * i);
  });
}

// Click handler for follow-up suggestion chips
function askFollowup(question) {
  const ta = document.getElementById('consumer-input');
  ta.value = question;
  autoResize(ta);
  sendConsumerMessage();
}

// Strip [N] citation markers, the boilerplate disclaimer, and the
// "You might also want to know:" follow-up section before pushing the
// assistant response into chatHistory. The backend will see clean prose.
function pushAssistantToHistory(response) {
  let text = response;
  // Drop the follow-up section (same regex as the main parser, broadened)
  const followupMarker = /(?:\n|^)\s*(?:\*+|#+)?\s*(?:you\s+might\s+also\s+want\s+to\s+know|follow[-\s]?up\s+questions|suggested\s+questions|related\s+questions)\s*:?\s*\*?\*?[\s\S]*$/i;
  text = text.replace(followupMarker, '');
  // Drop the boilerplate disclaimer
  text = text.replace(/this is for information only[^\n]*/gi, '');
  // Drop [N] citation markers
  text = text.replace(/\[\d+\]/g, '');
  // Collapse extra whitespace
  text = text.replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  if (text.length > 5) {
    chatHistory.push({ role: 'assistant', content: text });
  }
}

function scrollChat() {
  const msgs = document.getElementById('chat-messages');
  msgs.scrollTop = msgs.scrollHeight;
}

// ── Professional search ──────────────────────────────────────────────────────
function handleProKey(e) { if (e.key === 'Enter') sendProQuery(); }

function resetFilters() {
  document.querySelectorAll('.pub-type').forEach(c => c.checked = false);
  document.getElementById('year-from').value = '';
  document.getElementById('year-to').value = '';
  document.querySelector('[name=strategy][value=full]').checked = true;
  document.getElementById('pro-provider').value = 'extractive';
  document.getElementById('topk-slider').value = 10;
  document.getElementById('topk-val').textContent = '10';
}

async function sendProQuery() {
  const input = document.getElementById('pro-input');
  const query = input.value.trim();
  if (!query) return;

  const btn = document.querySelector('.pro-search-btn');
  btn.disabled = true;
  btn.textContent = 'Searching…';

  const pubTypes = [...document.querySelectorAll('.pub-type:checked')].map(c => c.value);
  const yearFrom = document.getElementById('year-from').value;
  const yearTo   = document.getElementById('year-to').value;
  const strategy = document.querySelector('[name=strategy]:checked').value;
  const provider = document.getElementById('pro-provider').value;
  const topK     = parseInt(document.getElementById('topk-slider').value, 10);

  const payload = {
    query,
    filters: {
      ...(pubTypes.length && { publication_types: pubTypes }),
      ...(yearFrom        && { pub_year_from: parseInt(yearFrom, 10) }),
      ...(yearTo          && { pub_year_to:   parseInt(yearTo,   10) }),
    },
    options: { top_k: topK, retrieval_strategy: strategy, llm_provider: provider },
  };

  const resultsEl    = document.getElementById('pro-results');
  const responseBox  = document.getElementById('pro-response-box');
  resultsEl.innerHTML = renderSkeletons(3);
  responseBox.hidden = true;

  try {
    const r = await apiFetch(`${API}/api/v1/query`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      resultsEl.innerHTML = `<div class="empty-state"><p>Error: ${_formatErr(err, r)}</p></div>`;
      console.error('[pro-search] HTTP', r.status, err);
      return;
    }
    renderProResults(await r.json());
  } catch {
    resultsEl.innerHTML = `<div class="empty-state"><p>Could not reach the API.</p></div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Search';
  }
}

function renderProResults(data) {
  const responseBox  = document.getElementById('pro-response-box');
  const responseText = document.getElementById('pro-response-text');
  const modelBadge   = document.getElementById('pro-model-badge');
  const latencyBadge = document.getElementById('pro-latency');
  const resultsEl    = document.getElementById('pro-results');

  if (data.response && data.metadata?.model_used !== 'extractive') {
    responseBox.hidden = false;
    responseText.textContent = data.response;
    modelBadge.textContent   = data.metadata?.model_used ?? '';
    latencyBadge.textContent = data.metadata?.latency_ms ? `${data.metadata.latency_ms}ms` : '';
  }

  const citations = data.citations ?? [];
  if (!citations.length) {
    resultsEl.innerHTML = `<div class="empty-state"><p>No results found. Try broadening your query.</p></div>`;
    return;
  }
  resultsEl.innerHTML = citations.map(renderResultCard).join('');
}

function renderResultCard(c, i) {
  const pubTypes = (c.publication_types ?? []);
  const badges = pubTypes.map(pt => {
    let cls = 'other-badge';
    if (pt.includes('Randomized')) cls = 'rct-badge';
    else if (pt.includes('Meta')) cls = 'meta-badge';
    else if (pt.includes('Review') || pt.includes('Systematic')) cls = 'review-badge';
    return `<span class="pub-type-badge ${cls}">${pt}</span>`;
  }).join(' ');

  const score   = c.relevance_score ?? 0;
  const passage = c.chunk?.text ?? c.passage ?? '';
  const pmidUrl = `https://pubmed.ncbi.nlm.nih.gov/${c.pmid}/`;

  return `
    <div class="result-card">
      <div class="result-header">
        <div class="result-rank">${i + 1}</div>
        <div class="result-title">
          <a href="${pmidUrl}" target="_blank" rel="noopener">${escapeHtml(c.title ?? 'Untitled')}</a>
        </div>
      </div>
      <div class="result-meta">
        ${c.authors ? `<span class="meta-authors">${escapeHtml(c.authors)}</span>` : ''}
        ${c.authors && (c.journal || c.year) ? '<span class="meta-sep">·</span>' : ''}
        ${c.journal ? `<span class="meta-journal">${escapeHtml(c.journal)}</span>` : ''}
        ${c.year    ? `<span class="meta-year">(${c.year})</span>` : ''}
        ${c.pmid    ? `<span class="meta-sep">·</span><span class="meta-pmid">PMID ${c.pmid}</span>` : ''}
        ${badges}
      </div>
      ${passage ? `<div class="result-passage">${escapeHtml(passage.slice(0, 320))}${passage.length > 320 ? '…' : ''}</div>` : ''}
      <div class="result-footer">
        <div class="score-bar-wrap">
          <div class="score-bar"><div class="score-fill" style="width:${Math.round(score * 100)}%"></div></div>
          <span class="score-text">${(score * 100).toFixed(0)}% match</span>
        </div>
        <button class="detail-btn" onclick="openModal(${JSON.stringify(c).replace(/"/g, '&quot;')})">Details</button>
      </div>
    </div>`;
}

function renderSkeletons(n) {
  return Array.from({ length: n }, () => `
    <div class="result-card">
      <div class="skeleton" style="height:14px;width:55%;margin-bottom:10px"></div>
      <div class="skeleton" style="height:11px;width:80%;margin-bottom:7px"></div>
      <div class="skeleton" style="height:11px;width:35%"></div>
    </div>`).join('');
}

// ── Citation modal ──────────────────────────────────────────────────────────
function openModal(citation) {
  const modal   = document.getElementById('citation-modal');
  const content = document.getElementById('modal-content');
  const pmidUrl = `https://pubmed.ncbi.nlm.nih.gov/${citation.pmid}/`;

  const pubTypes = (citation.publication_types ?? []).map(pt =>
    `<span class="modal-meta-tag">${escapeHtml(pt)}</span>`
  ).join('');

  const passage = citation.chunk?.text ?? citation.passage ?? '';

  content.innerHTML = `
    <a class="modal-pmid-link" href="${pmidUrl}" target="_blank" rel="noopener">
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M4.5 2H2C1.4 2 1 2.4 1 3V10C1 10.6 1.4 11 2 11H9C9.6 11 10 10.6 10 10V7.5M7 1H11M11 1V5M11 1L5 7" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
      PMID ${citation.pmid} — View on PubMed
    </a>
    <div class="modal-title">${escapeHtml(citation.title ?? 'Untitled')}</div>
    <div class="modal-authors">${escapeHtml(citation.authors ?? '')} · ${escapeHtml(citation.journal ?? '')} ${citation.year ? `(${citation.year})` : ''}</div>
    ${citation.abstract ? `<div class="modal-abstract">${escapeHtml(citation.abstract)}</div>` : ''}
    ${passage ? `<div class="modal-passage-label">Cited passage</div><div class="modal-passage">${escapeHtml(passage)}</div>` : ''}
    <div class="modal-meta-row">${pubTypes}</div>`;

  modal.hidden = false;
}

function closeModal(e) {
  if (e.target === document.getElementById('citation-modal')) {
    document.getElementById('citation-modal').hidden = true;
  }
}

// ── Pull-to-dismiss the citation modal on phones (Phase 6) ──────────────
// Touch starts in the top ~60 px of the .modal-card (where the drag
// handle lives) initiate a drag. The card follows the finger via
// translateY; releasing past 120 px closes the modal.
function _bindModalSwipe() {
  const card = document.getElementById('citation-modal-card');
  if (!card || card._swipeBound) return;
  card._swipeBound = true;
  let startY = 0, dy = 0, dragging = false;

  card.addEventListener('touchstart', (e) => {
    const r = card.getBoundingClientRect();
    // Only start the drag from the handle area (top 60 px). Below that,
    // touches scroll the modal content.
    if (e.touches[0].clientY - r.top > 60) return;
    dragging = true;
    startY = e.touches[0].clientY;
    dy = 0;
    card.style.transition = 'none';
  }, { passive: true });

  card.addEventListener('touchmove', (e) => {
    if (!dragging) return;
    dy = Math.max(0, e.touches[0].clientY - startY);
    card.style.transform = `translateY(${dy}px)`;
    // Fade the backdrop opacity as the card moves.
    const modal = document.getElementById('citation-modal');
    if (modal) modal.style.opacity = String(Math.max(0.3, 1 - dy / 400));
  }, { passive: true });

  card.addEventListener('touchend', () => {
    if (!dragging) return;
    dragging = false;
    card.style.transition = 'transform 200ms ease-out';
    const modal = document.getElementById('citation-modal');
    if (dy > 120) {
      if (modal) modal.hidden = true;
    }
    card.style.transform = '';
    if (modal) modal.style.opacity = '';
    dy = 0;
  });
}
// Bind once on script load — modal element exists at DOM-ready time.
if (document.readyState !== 'loading') _bindModalSwipe();
else document.addEventListener('DOMContentLoaded', _bindModalSwipe);

// ── Duplicate-label detection → Merge-button hi-vis (Phase 6) ──────────
// Called after each successful _extractGraph. Cheap O(n²) Jaccard on
// lowercased token sets — at our typical 10–40-node graph the work
// is negligible (<1 ms). Toggles .has-duplicates on the Merge button so
// users get a visual nudge when the graph has obvious dupes.
function _detectAndFlagDuplicates() {
  const btn = document.getElementById('graph-merge-btn');
  if (!btn) return;
  let labels;
  try { labels = KGraph.exportJSON().nodes.map((n) => n.label); }
  catch (_) { return; }
  if (!labels || labels.length < 2) {
    btn.classList.remove('has-duplicates');
    return;
  }
  // Token sets for Jaccard
  const tokens = labels.map((l) => new Set(
    l.toLowerCase().replace(/[^a-z0-9\s-]/g, ' ').split(/\s+/).filter(Boolean)
  ));
  const jacc = (a, b) => {
    if (a.size === 0 || b.size === 0) return 0;
    let inter = 0;
    for (const t of a) if (b.has(t)) inter++;
    return inter / (a.size + b.size - inter);
  };
  let found = false;
  outer: for (let i = 0; i < tokens.length; i++) {
    for (let j = i + 1; j < tokens.length; j++) {
      if (jacc(tokens[i], tokens[j]) >= 0.7) { found = true; break outer; }
    }
  }
  btn.classList.toggle('has-duplicates', found);
}

// Read an SSE response that emits a single final `{ok, payload|error}`
// event, with `: keepalive` comments in between. Used by /graph_extract and
// /graph_dedup. Returns the payload on success, throws on backend error,
// returns null if the stream ends without a final event.
// Start a long-running graph job and poll until done. Each HTTP round-trip
// stays well under any tunnel cap, so the LLM can run as long as it needs.
//   startUrl: POST → returns {job_id}
//   jobUrlPrefix: GET ${jobUrlPrefix}/${job_id} → returns {status, payload?, error?}
// Returns the final `payload` dict, throws on backend error or polling timeout.
async function _runGraphJob({ startUrl, jobUrlPrefix, startBody, maxWaitMs = 600000, pollMs = 1500, onProgress = null }) {
  const r = await apiFetch(startUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(startBody),
    signal: AbortSignal.timeout(15000),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(`start failed: HTTP ${r.status} ${err.detail || ''}`);
  }
  const { job_id } = await r.json();
  if (!job_id) throw new Error('start response missing job_id');

  // Poll the first time almost immediately — a fast LLM call (Haiku or NIM
  // on a small graph) routinely finishes in <500 ms, so a 1500 ms sleep
  // before the first poll wastes that much wall clock on the user. After
  // the first miss, fall back to the configured pollMs cadence.
  // Per-fetch AbortSignal.timeout(8000) guards against a hung tunnel: a
  // single dropped response can no longer stall the overall deadline.
  // The server publishes a heartbeat with `elapsed_s` every 500 ms; we
  // pass each pending snapshot to onProgress so the caller can surface
  // "still working… Ns" UI.
  const deadline = Date.now() + maxWaitMs;
  let firstPoll = true;
  while (Date.now() < deadline) {
    await new Promise(res => setTimeout(res, firstPoll ? 150 : pollMs));
    firstPoll = false;
    let pr;
    try {
      pr = await apiFetch(`${jobUrlPrefix}/${encodeURIComponent(job_id)}`, {
        signal: AbortSignal.timeout(8000),
      });
    } catch (e) {
      // Transient network blip or per-fetch timeout — keep polling.
      continue;
    }
    if (!pr.ok) {
      if (pr.status === 404) throw new Error('job expired or unknown');
      continue;  // transient 5xx — keep polling
    }
    const status = await pr.json().catch(() => ({}));
    if (status.status === 'done') return status.payload;
    if (status.status === 'error') throw new Error(status.error || 'graph job failed');
    // status === 'pending' → keep polling. Forward the heartbeat snapshot
    // (elapsed_s, stage, …) to the caller for live UI updates.
    if (onProgress && typeof status.elapsed_s === 'number') {
      try { onProgress(status); } catch (_) { /* never let a UI bug break polling */ }
    }
  }
  throw new Error(`graph job timed out after ${Math.round(maxWaitMs / 1000)}s polling`);
}

// ── Utilities ────────────────────────────────────────────────────────────────
function _downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function _timestamp() {
  return new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Drain any onclick="switchMode(...)" calls that fired against the inline
// stub before app.js parsed. The real switchMode (function declaration above)
// is hoisted to window at script start, so by the time we reach this line
// it has already replaced the stub.
if (window.__pendingMode) {
  const mode = window.__pendingMode;
  delete window.__pendingMode;
  switchMode(mode);
}
