'use strict';

const API = '';

// ── View switching ──────────────────────────────────────────────────────────
function switchMode(mode) {
  ['landing', 'consumer', 'professional'].forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (el) el.classList.toggle('active', v === mode);
  });
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

// ── Consumer chat ────────────────────────────────────────────────────────────
let activeTopic = 'lupus';

function setTopic(t) {
  activeTopic = t;
  document.querySelectorAll('.chip').forEach(c =>
    c.classList.toggle('active-chip', c.getAttribute('onclick').includes(`'${t}'`))
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
        options: { top_k: 8, retrieval_strategy: 'bm25', llm_provider: provider, plain_language: simple },
      }),
    });

    removeTypingBubble(typing);
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      appendAIBubble(`Sorry, I couldn't get a response. ${err.detail ?? r.statusText}`, []);
      return;
    }
    const data = await r.json();
    appendAIBubble(data.response ?? 'No response returned.', data.citations ?? []);
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

  d.innerHTML = `
    <div class="msg-avatar">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 2C5.24 2 3 4.24 3 7C3 9.76 5.24 12 8 12C10.76 12 13 9.76 13 7" stroke="white" stroke-width="1.5" stroke-linecap="round"/><path d="M7 5H9M8 3V9" stroke="white" stroke-width="1.5" stroke-linecap="round"/></svg>
    </div>
    <div class="msg-content"><div>${linked}</div>${citeHtml}</div>`;
  msgs.appendChild(d);
  scrollChat();
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
