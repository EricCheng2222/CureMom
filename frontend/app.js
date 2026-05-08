'use strict';

const API = '';

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

  const ollama = status.providers?.ollama || {};
  const installed = ollama.installed_models || [];
  const defaultOllama = ollama.model;
  const claude = status.providers?.claude || {};
  const openai = status.providers?.openai || {};

  for (const id of ['consumer-provider', 'pro-provider']) {
    const sel = document.getElementById(id);
    if (!sel) continue;
    // Strip everything except the static "Extractive" option
    [...sel.options].forEach(o => { if (o.value !== 'extractive') o.remove(); });

    if (installed.length) {
      const optgroup = document.createElement('optgroup');
      optgroup.label = ollama.available ? 'Ollama (local)' : 'Ollama (offline?)';
      installed.forEach(m => {
        const opt = document.createElement('option');
        opt.value = `ollama/${m}`;
        opt.textContent = m;
        if (m === defaultOllama || m === `${defaultOllama}:latest`) {
          opt.textContent += '  (default)';
          opt.dataset.isDefault = '1';
        }
        optgroup.appendChild(opt);
      });
      sel.appendChild(optgroup);
    } else if (ollama.endpoint) {
      const opt = document.createElement('option');
      opt.value = 'ollama'; opt.disabled = true;
      opt.textContent = 'Ollama — no models installed';
      sel.appendChild(opt);
    }

    if (claude.available) {
      const opt = document.createElement('option');
      opt.value = 'claude';
      opt.textContent = `Claude (${claude.model})`;
      sel.appendChild(opt);
    }
    if (openai.available) {
      const opt = document.createElement('option');
      opt.value = 'openai';
      opt.textContent = `OpenAI (${openai.model})`;
      sel.appendChild(opt);
    }

    // Auto-select the configured default Ollama model if there is one
    if (status.configured_provider === 'ollama' && installed.length) {
      const def = [...sel.options].find(o => o.dataset.isDefault === '1');
      if (def) sel.value = def.value;
    }
  }
  console.log('[providers] dropdown populated with', installed.length, 'Ollama models');
}
populateProviderDropdowns().catch(err => {
  console.error('[providers] populate failed:', err);
});

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
  const startOpen = savedPref === null ? true : savedPref === '1';
  if (startOpen) panel.classList.remove('collapsed');
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
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `curemom-graph-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
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
  if (!_graphInitialized) {
    console.log('[KGraph] merge skipped — graph not initialized');
    return;
  }
  const { nodes } = KGraph.size();
  if (nodes < 2) {
    console.log('[KGraph] merge skipped — need at least 2 nodes');
    return;
  }
  const labels = KGraph.exportJSON().nodes.map(n => n.label);
  const provider = document.getElementById('consumer-provider')?.value || null;
  const btn = document.getElementById('graph-merge-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Merging…'; }
  console.log('[KGraph] POST /api/v1/graph_dedup with', labels.length, 'labels, provider:', provider);
  try {
    const r = await fetch(`${API}/api/v1/graph_dedup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ labels, llm_provider: provider }),
    });
    if (!r.ok) {
      console.warn('[KGraph] /graph_dedup HTTP', r.status);
      return;
    }
    const payload = await r.json();
    const groups = payload.groups || [];
    console.log('[KGraph] received', groups.length, 'merge groups:', groups);
    const before = KGraph.size().nodes;
    const result = KGraph.applyMergeGroups(groups);
    const after = KGraph.size().nodes;
    const status = result.groupsApplied > 0
      ? `Merged ${result.groupsApplied} group${result.groupsApplied !== 1 ? 's' : ''} (${before} → ${after} nodes)`
      : 'No duplicates found';
    _refreshGraphChrome(status);
    // Revert to default chrome after a few seconds.
    setTimeout(() => _refreshGraphChrome(), 4000);
  } catch (err) {
    console.error('[KGraph] dedup failed:', err);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Merge'; }
  }
}

function _showGraphSpinner(show) {
  const sp = document.getElementById('graph-spinner');
  if (sp) sp.hidden = !show;
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

async function _extractGraph(query, response, citations) {
  console.log('[KGraph] _extractGraph called — query:', query.slice(0, 60), '| panel open:', _isGraphPanelOpen());
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
  try {
    const r = await fetch(`${API}/api/v1/graph_extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, answer: cleanAnswer, chunks, llm_provider: provider }),
    });
    if (!r.ok) {
      console.warn('[KGraph] /graph_extract HTTP', r.status);
      return;
    }
    const payload = await r.json();
    console.log('[KGraph] received payload:', (payload.nodes || []).length, 'nodes,', (payload.edges || []).length, 'edges');
    if (payload.error) {
      console.error('[KGraph] backend reported error:', payload.error);
    }
    if (!payload.nodes?.length && !payload.edges?.length) {
      if (payload.error) {
        console.warn('[KGraph] empty payload due to error above. Try a shorter question, or set OLLAMA_GRAPH_MODEL to a faster model.');
      } else {
        console.warn('[KGraph] empty payload — LLM produced no grounded entities/relations for this answer');
      }
      return;
    }
    const result = KGraph.merge(payload);
    console.log('[KGraph] merged —', result.addedNodes.length, 'new nodes,', result.addedEdges.length, 'new edges');
    _refreshGraphChrome();
    // If the panel is collapsed but we have new content, gently nudge it open
    // so the user sees the result.
    if (!_isGraphPanelOpen() && (result.addedNodes.length || result.addedEdges.length)) {
      console.log('[KGraph] auto-opening panel — first content arrived');
      toggleGraphPanel();
    }
  } catch (err) {
    console.error('[KGraph] /graph_extract fetch failed:', err);
  } finally {
    _showGraphSpinner(false);
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

  pop.innerHTML = `
    <button class="popover-close-x" aria-label="Close">×</button>
    <div class="popover-header">
      <div class="popover-label">${escapeHtml(nodePayload.label)}</div>
      <div class="popover-type-chip">${escapeHtml(_humanizeType(nodePayload.type))}</div>
    </div>
    ${citePills}
    <div class="popover-actions">
      <button class="popover-ask-btn">Ask about this</button>
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
    const ta = document.getElementById('consumer-input');
    if (!ta) return;
    ta.value = `Tell me more about ${nodePayload.label} in this context.`;
    autoResize(ta);
    ta.focus();
    _hidePopover();
  });
  pop.querySelector('.popover-remove-btn')?.addEventListener('click', () => {
    if (!_graphInitialized) return;
    KGraph.removeNode(nodePayload.id);
    _refreshGraphChrome();
    _hidePopover();
  });
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

  try {
    const provider = document.getElementById('consumer-provider').value;
    const simple   = document.getElementById('consumer-simple').checked;

    const r = await fetch(`${API}/api/v1/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        options: { top_k: 12, retrieval_strategy: 'full', llm_provider: provider, plain_language: simple },
        history: chatHistory.slice(-MAX_HISTORY_TURNS * 2),  // last N user+assistant pairs
      }),
    });

    removeTypingBubble(typing);
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      appendAIBubble(`Sorry, I couldn't get a response. ${err.detail ?? r.statusText}`, []);
      return;
    }
    const data = await r.json();
    const response = data.response ?? 'No response returned.';
    appendAIBubble(response, data.citations ?? []);
    // Push BOTH turns to history. The user message was already pushed in
    // appendUserBubble; here we add the assistant response as clean prose
    // (strip [N] markers + disclaimer + follow-up section).
    pushAssistantToHistory(response);
    // Fire-and-forget: extract knowledge-graph nodes/edges in the background
    // and merge into the side panel. Won't block the UI.
    _extractGraph(query, response, data.citations ?? []);
  } catch {
    removeTypingBubble(typing);
    appendAIBubble('Could not reach the API. Make sure the server is running.', []);
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
    </div>`;
  msgs.appendChild(d);
  scrollChat();
  return d;
}

function removeTypingBubble(el) { el?.remove(); }

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
    <div class="msg-content"><div>${linked}</div>${citeHtml}${followupHtml}</div>`;
  msgs.appendChild(d);
  scrollChat();
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
    const r = await fetch(`${API}/api/v1/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      resultsEl.innerHTML = `<div class="empty-state"><p>Error: ${err.detail ?? r.statusText}</p></div>`;
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

// ── Utilities ────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
