'use strict';

// ── Config ──────────────────────────────────────────────────────────────────
// In production, replace with the actual server URL. During dev, FastAPI
// serves both the API and this frontend from the same origin.
const API = '';  // same-origin; change to 'http://localhost:8000' if serving separately

// ── View switching ──────────────────────────────────────────────────────────
function switchMode(mode) {
  ['landing', 'consumer', 'professional'].forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (!el) return;
    el.hidden = (v !== mode);
    el.classList.toggle('active', v === mode);
  });
}
switchMode('landing');

// ── Status check ────────────────────────────────────────────────────────────
async function checkStatus() {
  const dot  = document.querySelector('.status-dot');
  const text = document.getElementById('status-text');
  try {
    const r = await fetch(`${API}/api/v1/stats`, { signal: AbortSignal.timeout(4000) });
    if (r.ok) {
      const d = await r.json();
      dot.className = 'status-dot ok';
      const n = d.papers_count ?? d.total_papers ?? '?';
      text.textContent = `${n.toLocaleString()} papers indexed`;
    } else {
      dot.className = 'status-dot warn';
      text.textContent = 'API reachable, limited data';
    }
  } catch {
    dot.className = 'status-dot err';
    text.textContent = 'API offline — start with: uvicorn src.api.main:app';
  }
}
checkStatus();

// ── Consumer chat ────────────────────────────────────────────────────────────
let activeTopic = 'lupus';

function setTopic(t) {
  activeTopic = t;
  document.querySelectorAll('.topic-chip').forEach(c =>
    c.classList.toggle('active-topic', c.getAttribute('onclick').includes(`'${t}'`))
  );
}

function fillExample(btn) {
  const ta = document.getElementById('consumer-input');
  ta.value = btn.textContent.trim();
  autoResize(ta);
  ta.focus();
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendConsumerMessage(); }
}

async function sendConsumerMessage() {
  const ta = document.getElementById('consumer-input');
  const query = ta.value.trim();
  if (!query) return;

  appendUserBubble(query);
  ta.value = '';
  autoResize(ta);

  const btn = document.getElementById('consumer-send');
  btn.disabled = true;
  const typing = appendTypingBubble();

  try {
    const provider = document.getElementById('consumer-provider').value;
    const simple   = document.getElementById('consumer-simple').checked;

    const payload = {
      query,
      options: {
        top_k: 8,
        retrieval_strategy: 'bm25',
        llm_provider: provider,
        plain_language: simple,
      },
    };

    const r = await fetch(`${API}/api/v1/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    removeTypingBubble(typing);

    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      appendAIBubble(
        `Sorry, I couldn't get a response right now. ${err.detail ?? r.statusText}`,
        []
      );
      return;
    }

    const data = await r.json();
    appendAIBubble(data.response ?? 'No response returned.', data.citations ?? []);
  } catch (e) {
    removeTypingBubble(typing);
    appendAIBubble(
      'Could not reach the API. Make sure the server is running: `uvicorn src.api.main:app --reload`',
      []
    );
  } finally {
    btn.disabled = false;
  }
}

function appendUserBubble(text) {
  const msgs = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'message user-message';
  d.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
  msgs.appendChild(d);
  scrollChat();
}

function appendTypingBubble() {
  const msgs = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'message ai-message typing-indicator';
  d.innerHTML = `
    <div class="message-avatar">
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="10" fill="#6366F1"/></svg>
    </div>
    <div class="message-bubble">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>`;
  msgs.appendChild(d);
  scrollChat();
  return d;
}

function removeTypingBubble(el) {
  el?.remove();
}

function appendAIBubble(text, citations) {
  const msgs = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'message ai-message';

  // Linkify [N] citation markers
  const linked = text.replace(/\[(\d+)\]/g, (_, n) => {
    const c = citations[parseInt(n, 10) - 1];
    return c
      ? `<button class="citation-ref" onclick="openModal(${JSON.stringify(c).replace(/"/g, '&quot;')})">[${n}]</button>`
      : `[${n}]`;
  });

  let citeHtml = '';
  if (citations.length) {
    const pills = citations.slice(0, 5).map((c, i) => `
      <button class="citation-pill" onclick="openModal(${JSON.stringify(c).replace(/"/g, '&quot;')})">
        <div class="citation-index">${i + 1}</div>
        <div class="citation-meta">
          <strong>${escapeHtml(c.title ?? 'Unknown title')}</strong><br>
          ${escapeHtml(c.authors ?? '')} ${c.year ? `(${c.year})` : ''} ·
          ${escapeHtml(c.journal ?? '')}
        </div>
      </button>`).join('');
    citeHtml = `
      <button class="citations-toggle" onclick="this.nextElementSibling.hidden=!this.nextElementSibling.hidden">
        📄 ${citations.length} source${citations.length > 1 ? 's' : ''}
      </button>
      <div class="citations-block" hidden>${pills}</div>`;
  }

  d.innerHTML = `
    <div class="message-avatar">
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="10" fill="#6366F1"/>
        <path d="M10 4C7.2 4 5 6.2 5 9C5 11.8 7.2 14 10 14C12.8 14 15 11.8 15 9" stroke="white" stroke-width="1.5"/>
        <path d="M9 7H11M10 5V11" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
    </div>
    <div class="message-bubble">
      <div>${linked}</div>
      ${citeHtml}
    </div>`;
  msgs.appendChild(d);
  scrollChat();
}

function scrollChat() {
  const msgs = document.getElementById('chat-messages');
  msgs.scrollTop = msgs.scrollHeight;
}

// ── Professional search ──────────────────────────────────────────────────────
function handleProKey(e) {
  if (e.key === 'Enter') sendProQuery();
}

function resetFilters() {
  document.querySelectorAll('.pub-type').forEach(c => c.checked = false);
  document.getElementById('year-from').value = '';
  document.getElementById('year-to').value = '';
  document.querySelector('[name=strategy][value=bm25]').checked = true;
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
      ...(yearFrom && { pub_year_from: parseInt(yearFrom, 10) }),
      ...(yearTo   && { pub_year_to:   parseInt(yearTo,   10) }),
    },
    options: {
      top_k: topK,
      retrieval_strategy: strategy,
      llm_provider: provider,
    },
  };

  const resultsEl = document.getElementById('pro-results');
  const responseBox = document.getElementById('pro-response-box');
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
      resultsEl.innerHTML = `<div class="pro-empty-state"><p>Error: ${err.detail ?? r.statusText}</p></div>`;
      return;
    }

    const data = await r.json();
    renderProResults(data);
  } catch (e) {
    resultsEl.innerHTML = `<div class="pro-empty-state"><p>Could not reach the API. Is the server running?</p></div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Search';
  }
}

function renderProResults(data) {
  const responseBox = document.getElementById('pro-response-box');
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
    resultsEl.innerHTML = `<div class="pro-empty-state"><p>No results found. Try broadening your query or removing filters.</p></div>`;
    return;
  }

  resultsEl.innerHTML = citations.map((c, i) => renderResultCard(c, i)).join('');
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
        ${c.authors ? `<span>👤 ${escapeHtml(c.authors)}</span>` : ''}
        ${c.journal  ? `<span>📰 ${escapeHtml(c.journal)}</span>` : ''}
        ${c.year     ? `<span>📅 ${c.year}</span>` : ''}
        ${c.pmid     ? `<span>🔗 PMID ${c.pmid}</span>` : ''}
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
      <div class="skeleton" style="height:16px;width:60%;margin-bottom:10px"></div>
      <div class="skeleton" style="height:12px;width:85%;margin-bottom:8px"></div>
      <div class="skeleton" style="height:12px;width:40%"></div>
    </div>`).join('');
}

// ── Citation detail modal ────────────────────────────────────────────────────
function openModal(citation) {
  const modal = document.getElementById('citation-modal');
  const content = document.getElementById('modal-content');
  const pmidUrl = `https://pubmed.ncbi.nlm.nih.gov/${citation.pmid}/`;

  const pubTypes = (citation.publication_types ?? []).map(pt =>
    `<span class="modal-meta-tag">${escapeHtml(pt)}</span>`
  ).join('');

  const passage = citation.chunk?.text ?? citation.passage ?? '';

  content.innerHTML = `
    <a class="modal-pmid-link" href="${pmidUrl}" target="_blank" rel="noopener">
      <svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M5 2H2C1.4 2 1 2.4 1 3V11C1 11.6 1.4 12 2 12H10C10.6 12 11 11.6 11 11V8M8 1H12M12 1V5M12 1L5.5 7.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
      PMID ${citation.pmid} — View on PubMed
    </a>
    <div class="modal-title">${escapeHtml(citation.title ?? 'Untitled')}</div>
    <div class="modal-authors">${escapeHtml(citation.authors ?? '')} · ${escapeHtml(citation.journal ?? '')} ${citation.year ? `(${citation.year})` : ''}</div>
    ${citation.abstract ? `<div class="modal-abstract">${escapeHtml(citation.abstract)}</div>` : ''}
    ${passage ? `
      <div class="modal-passage-label">Cited passage</div>
      <div class="modal-passage">${escapeHtml(passage)}</div>` : ''}
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

// Citation ref button style (inline, so it inherits bubble context)
const refStyle = document.createElement('style');
refStyle.textContent = `
  .citation-ref {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 18px; border-radius: 4px;
    background: #EEF2FF; color: #6366F1; font-size: 11px; font-weight: 700;
    border: 1px solid #C7D2FE; cursor: pointer; margin: 0 1px;
    vertical-align: middle; transition: background .15s;
  }
  .citation-ref:hover { background: #C7D2FE; }
`;
document.head.appendChild(refStyle);
