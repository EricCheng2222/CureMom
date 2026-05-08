/* KGraph — session-local knowledge graph for the chat panel.
 *
 * Public API on the global `KGraph` object:
 *   KGraph.init(canvasEl)                 -> initialize Cytoscape on a div
 *   KGraph.merge(payload)                 -> merge a per-turn { nodes, edges } from /api/v1/graph_extract
 *   KGraph.clear()                        -> wipe the canvas
 *   KGraph.exportJSON()                   -> { nodes: [...], edges: [...] } of current state
 *   KGraph.size()                         -> { nodes, edges }
 *   KGraph.onNodeClick(handler)           -> register a click handler; receives { id, label, type, citations, position }
 *
 * Storage: in-memory Maps. Resets on page reload.
 */
(function () {
  // Cytoscape can't resolve `var(--...)` in style values — it doesn't
  // walk computed styles. Read each CSS variable once at module load
  // and hand Cytoscape resolved hex strings. Fallbacks match the
  // values declared in style.css so the graph still works if the
  // stylesheet hasn't loaded yet.
  function _readVar(name, fallback) {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fallback;
    } catch (_) {
      return fallback;
    }
  }
  const NODE_TYPE_COLORS = {
    CHEMICAL:             _readVar('--node-chemical', '#34D399'),
    DISEASE:              _readVar('--node-disease',  '#F87171'),
    GENE_OR_GENE_PRODUCT: _readVar('--node-gene',     '#818CF8'),
    ANATOMY:              _readVar('--node-anatomy',  '#FBBF24'),
    SYMPTOM:              _readVar('--node-symptom',  '#FB7185'),
    PROCEDURE:            _readVar('--node-procedure','#A78BFA'),
    CELL_TYPE:            _readVar('--node-anatomy',  '#FBBF24'),
    ORGANISM:             _readVar('--node-default',  '#94A3B8'),
    OTHER:                _readVar('--node-default',  '#94A3B8'),
  };

  const state = {
    nodes: new Map(),   // id -> { id, label, type, citations: [], kb_id }
    edges: new Map(),   // id -> { id, source, target, predicate, citations: [] }
  };

  let cy = null;
  let nodeClickHandlers = [];

  function init(container) {
    if (typeof cytoscape === 'undefined') {
      console.error('[KGraph] cytoscape library NOT loaded — check that the CDN <script> in index.html resolved');
      return null;
    }
    console.log('[KGraph] cytoscape lib version:', cytoscape.version || '(unknown)');
    if (typeof cytoscape.use === 'function' && typeof window.cytoscapeFcose !== 'undefined') {
      try { cytoscape.use(window.cytoscapeFcose); console.log('[KGraph] fcose layout registered'); } catch (e) { console.log('[KGraph] fcose already registered'); }
    } else {
      console.warn('[KGraph] fcose layout NOT available — falling back to cose');
    }

    cy = cytoscape({
      container: container,
      elements: [],
      // Pan + zoom are on by default; we make wheel zoom feel snappier and
      // clamp the zoom range so users can't get lost.
      wheelSensitivity: 0.4,
      minZoom: 0.15,
      maxZoom: 5.0,
      userPanningEnabled: true,
      userZoomingEnabled: true,
      boxSelectionEnabled: false,
      autoungrabify: false,            // user can drag individual nodes
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele) => NODE_TYPE_COLORS[ele.data('type')] || NODE_TYPE_COLORS.OTHER,
            'label':            'data(label)',
            'color':            '#F1F5F9',
            'font-size':        '9px',
            'font-weight':      '600',
            'text-valign':      'bottom',
            'text-halign':      'center',
            'text-margin-y':    5,
            'text-outline-color': '#0A0E1A',
            'text-outline-width': 2,
            'text-wrap':        'wrap',
            'text-max-width':   '90px',
            'border-width':     1.5,
            'border-color':     'rgba(255,255,255,.22)',
            'width':            22,
            'height':           22,
            'transition-property': 'border-color, border-width, background-color, width, height',
            'transition-duration': '180ms',
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#FFFFFF',
            'border-width': 2.5,
            'width':  28,
            'height': 28,
          },
        },
        {
          selector: 'node.pulse',
          style: {
            'border-color': '#FFFFFF',
            'border-width': 4,
          },
        },
        {
          selector: 'node.dimmed',
          style: { 'opacity': 0.25 },
        },
        {
          selector: 'edge',
          style: {
            'curve-style':       'bezier',
            'width':             1.4,
            'line-color':        'rgba(255,255,255,.28)',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': 'rgba(255,255,255,.45)',
            'arrow-scale':       0.9,
            'label':             'data(predicate)',
            'font-size':         '8px',
            'font-weight':       '600',
            'color':             '#E2E8F0',
            'text-rotation':     'autorotate',
            'text-background-color': '#0A0E1A',
            'text-background-opacity': 0.92,
            'text-background-padding': '3px',
            'text-background-shape':   'roundrectangle',
            'text-border-color': 'rgba(255,255,255,.12)',
            'text-border-width': 1,
            'text-border-opacity': 1,
            'text-margin-y':     -2,
            'transition-property':'line-color, width, target-arrow-color',
            'transition-duration':'180ms',
          },
        },
        {
          selector: 'edge.highlighted',
          style: {
            'line-color':         '#818CF8',
            'target-arrow-color': '#818CF8',
            'width':              2.5,
          },
        },
        {
          selector: 'edge.pulse',
          style: {
            'line-color':         '#FFFFFF',
            'target-arrow-color': '#FFFFFF',
            'width':              2.5,
          },
        },
        {
          selector: 'edge.dimmed',
          style: { 'opacity': 0.15 },
        },
      ],
      layout: { name: 'preset' }, // we'll re-run fcose after merges
    });

    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      const data = node.data();
      const renderedPos = node.renderedPosition();
      // Highlight neighborhood
      cy.elements().removeClass('highlighted dimmed');
      const neighborhood = node.closedNeighborhood();
      cy.elements().not(neighborhood).addClass('dimmed');
      neighborhood.edges().addClass('highlighted');
      // Notify subscribers (popover lives in app.js)
      const stored = state.nodes.get(data.id);
      const payload = {
        id: data.id,
        label: data.label,
        type: data.type,
        citations: stored ? stored.citations.slice() : [],
        renderedPosition: renderedPos,
      };
      nodeClickHandlers.forEach((h) => { try { h(payload); } catch (e) { console.error(e); } });
    });

    cy.on('tap', (evt) => {
      // Background click clears highlight + tells listeners to close popover
      if (evt.target === cy) {
        cy.elements().removeClass('highlighted dimmed');
        nodeClickHandlers.forEach((h) => { try { h(null); } catch (_) {} });
      }
    });

    return cy;
  }

  function _layout() {
    if (!cy) return;
    if (!cy.elements().length) return;
    // Pick fcose if registered, otherwise fall back to built-in cose.
    const fcoseAvailable = !!(window.cytoscapeFcose);
    // Web-like force-directed layout. Randomize=true scatters nodes from
    // random start positions so chain-shaped graphs (A→B→C→D) get spread
    // out into 2D instead of collapsing onto a single line. quality='proof'
    // runs more iterations for better separation.
    cy.layout({
      name: fcoseAvailable ? 'fcose' : 'cose',
      animate: 'end',
      animationDuration: 500,
      randomize: true,                  // scatter from random; key for web-like layout
      quality: 'proof',                 // more iterations → better convergence
      nodeRepulsion: 24000,             // bigger = more spread
      idealEdgeLength: 140,
      edgeElasticity: 0.30,
      nestingFactor: 1.1,
      gravity: 0.08,                    // very low → no central pull, lets web open up
      gravityRange: 3.0,
      gravityCompound: 0.5,
      numIter: 4000,
      tile: true,
      tilingPaddingVertical: 30,
      tilingPaddingHorizontal: 30,
      uniformNodeDimensions: true,
      packComponents: true,             // disconnected components packed nicely side-by-side
      fit: true,
      padding: 40,
    }).run();
  }

  function merge(payload) {
    if (!cy) {
      console.warn('[KGraph] merge skipped — cy is null. init() may have failed.');
      return { addedNodes: [], addedEdges: [] };
    }
    if (!payload) return { addedNodes: [], addedEdges: [] };

    const incomingNodes = Array.isArray(payload.nodes) ? payload.nodes : [];
    const incomingEdges = Array.isArray(payload.edges) ? payload.edges : [];

    const addedNodeIds = [];
    const addedEdgeIds = [];

    for (const n of incomingNodes) {
      if (!n.id || !n.label) continue;
      const existing = state.nodes.get(n.id);
      if (existing) {
        // Union citations
        const seen = new Set(existing.citations);
        for (const c of (n.citations || [])) {
          if (!seen.has(c)) { existing.citations.push(c); seen.add(c); }
        }
        // Prefer richer label / type if upgrade is available
        if ((!existing.type || existing.type === 'OTHER') && n.type) existing.type = n.type;
        if (n.label && n.label.length > existing.label.length) existing.label = n.label;
        // Update Cytoscape display
        const cyNode = cy.getElementById(n.id);
        if (cyNode && cyNode.length) {
          cyNode.data('label', existing.label);
          cyNode.data('type', existing.type);
        }
      } else {
        const node = {
          id: n.id,
          label: n.label,
          type: n.type || 'OTHER',
          citations: Array.isArray(n.citations) ? n.citations.slice() : [],
          kb_id: n.kb_id || null,
        };
        state.nodes.set(node.id, node);
        cy.add({ group: 'nodes', data: { id: node.id, label: node.label, type: node.type } });
        addedNodeIds.push(node.id);
      }
    }

    for (const e of incomingEdges) {
      if (!e.id || !e.source || !e.target) continue;
      // Drop edges that reference unknown nodes (defensive)
      if (!state.nodes.has(e.source) || !state.nodes.has(e.target)) continue;
      const existing = state.edges.get(e.id);
      if (existing) {
        const seen = new Set(existing.citations);
        for (const c of (e.citations || [])) {
          if (!seen.has(c)) { existing.citations.push(c); seen.add(c); }
        }
      } else {
        const edge = {
          id: e.id,
          source: e.source,
          target: e.target,
          predicate: e.predicate || '',
          citations: Array.isArray(e.citations) ? e.citations.slice() : [],
        };
        state.edges.set(edge.id, edge);
        cy.add({
          group: 'edges',
          data: { id: edge.id, source: edge.source, target: edge.target, predicate: edge.predicate },
        });
        addedEdgeIds.push(edge.id);
      }
    }

    if (addedNodeIds.length || addedEdgeIds.length) {
      _layout();
    }

    // Pulse new elements briefly so the user can see what changed
    if (addedNodeIds.length) {
      const newNodes = cy.collection(addedNodeIds.map((id) => cy.getElementById(id)));
      newNodes.addClass('pulse');
      setTimeout(() => newNodes.removeClass('pulse'), 1800);
    }
    if (addedEdgeIds.length) {
      const newEdges = cy.collection(addedEdgeIds.map((id) => cy.getElementById(id)));
      newEdges.addClass('pulse');
      setTimeout(() => newEdges.removeClass('pulse'), 1800);
    }

    return { addedNodes: addedNodeIds, addedEdges: addedEdgeIds };
  }

  function clear() {
    state.nodes.clear();
    state.edges.clear();
    if (cy) cy.elements().remove();
  }

  function _slugify(text) {
    return (text || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'node';
  }

  function applyMergeGroups(groups) {
    // groups: [{canonical: string, members: [string, ...]}]
    // Each member's node folds into the canonical node; incident edges
    // are redirected; identical edges after redirect get unioned.
    if (!cy || !Array.isArray(groups) || !groups.length) {
      return { groupsApplied: 0, nodesRemoved: 0 };
    }

    // Map each existing node label (case-insensitive) -> id.
    const labelToId = new Map();
    for (const node of state.nodes.values()) {
      labelToId.set(node.label.toLowerCase(), node.id);
    }

    let groupsApplied = 0;
    let nodesRemoved = 0;

    for (const g of groups) {
      const canonicalLabel = (g.canonical || '').trim();
      const memberLabels = (g.members || []).filter(m => typeof m === 'string').map(m => m.trim());
      if (!canonicalLabel || memberLabels.length < 2) continue;

      // Resolve member labels to actual node ids in the current graph.
      const memberIds = [];
      const seenIds = new Set();
      for (const lbl of memberLabels) {
        const id = labelToId.get(lbl.toLowerCase());
        if (id && !seenIds.has(id)) {
          memberIds.push(id);
          seenIds.add(id);
        }
      }
      if (memberIds.length < 2) continue;   // nothing to merge in current graph

      // Pick canonical id: existing node with the canonical label if present,
      // otherwise the first member id (relabel to the canonical text).
      let canonicalId = labelToId.get(canonicalLabel.toLowerCase());
      if (!canonicalId || !memberIds.includes(canonicalId)) {
        canonicalId = memberIds[0];
      }

      // Union citations from all members into the canonical node, and adopt
      // the canonical label (LLM may have picked a different spelling).
      const canonicalNode = state.nodes.get(canonicalId);
      if (!canonicalNode) continue;
      const citeSet = new Set(canonicalNode.citations);
      for (const mid of memberIds) {
        if (mid === canonicalId) continue;
        const m = state.nodes.get(mid);
        if (!m) continue;
        for (const c of m.citations) if (!citeSet.has(c)) { canonicalNode.citations.push(c); citeSet.add(c); }
      }
      canonicalNode.label = canonicalLabel;
      const cyCanon = cy.getElementById(canonicalId);
      if (cyCanon && cyCanon.length) cyCanon.data('label', canonicalLabel);

      // Build a redirect map for the non-canonical members.
      const redirect = new Map();
      for (const mid of memberIds) {
        if (mid !== canonicalId) redirect.set(mid, canonicalId);
      }

      // Rewrite every edge in state.edges. Self-loops (source==target after
      // rewrite) are dropped. Identical edges (same s|p|o) are unioned.
      const newEdges = new Map();
      for (const edge of state.edges.values()) {
        const newSource = redirect.get(edge.source) || edge.source;
        const newTarget = redirect.get(edge.target) || edge.target;
        if (newSource === newTarget) continue;   // collapsed self-loop
        const newId = `${newSource}|${edge.predicate}|${newTarget}`;
        const existing = newEdges.get(newId);
        if (existing) {
          const seen = new Set(existing.citations);
          for (const c of edge.citations) if (!seen.has(c)) { existing.citations.push(c); seen.add(c); }
        } else {
          newEdges.set(newId, {
            id: newId,
            source: newSource,
            target: newTarget,
            predicate: edge.predicate,
            citations: edge.citations.slice(),
          });
        }
      }

      // Sync state.edges → drop ones not in newEdges, add new ones.
      const oldEdgeIds = new Set(state.edges.keys());
      const newEdgeIds = new Set(newEdges.keys());
      for (const oldId of oldEdgeIds) {
        if (!newEdgeIds.has(oldId)) {
          state.edges.delete(oldId);
          const cyEdge = cy.getElementById(oldId);
          if (cyEdge && cyEdge.length) cyEdge.remove();
        }
      }
      for (const [newId, edge] of newEdges) {
        const existing = state.edges.get(newId);
        if (existing) {
          existing.citations = edge.citations;
        } else {
          state.edges.set(newId, edge);
          cy.add({
            group: 'edges',
            data: { id: edge.id, source: edge.source, target: edge.target, predicate: edge.predicate },
          });
        }
      }

      // Remove non-canonical member nodes from state and canvas.
      for (const mid of memberIds) {
        if (mid === canonicalId) continue;
        state.nodes.delete(mid);
        labelToId.delete((state.nodes.get(mid)?.label || '').toLowerCase());
        const cyNode = cy.getElementById(mid);
        if (cyNode && cyNode.length) cyNode.remove();
        nodesRemoved++;
      }
      labelToId.set(canonicalLabel.toLowerCase(), canonicalId);
      groupsApplied++;
    }

    if (groupsApplied > 0) _layout();
    return { groupsApplied, nodesRemoved };
  }

  function removeNode(nodeId) {
    if (!nodeId) return false;
    if (!state.nodes.has(nodeId)) return false;
    // Remove every edge touching this node from session state.
    const edgesToDrop = [];
    for (const [eid, edge] of state.edges.entries()) {
      if (edge.source === nodeId || edge.target === nodeId) {
        edgesToDrop.push(eid);
      }
    }
    for (const eid of edgesToDrop) state.edges.delete(eid);
    state.nodes.delete(nodeId);
    if (cy) {
      const cyNode = cy.getElementById(nodeId);
      if (cyNode && cyNode.length) {
        // Cytoscape's remove() on a node also removes incident edges.
        cyNode.remove();
      }
    }
    // Re-layout if there are still elements; otherwise leave canvas empty.
    if (cy && cy.elements().length) _layout();
    return true;
  }

  function exportJSON() {
    return {
      nodes: Array.from(state.nodes.values()),
      edges: Array.from(state.edges.values()),
    };
  }

  function size() {
    return { nodes: state.nodes.size, edges: state.edges.size };
  }

  function onNodeClick(handler) {
    if (typeof handler === 'function') nodeClickHandlers.push(handler);
  }

  function zoomBy(factor) {
    if (!cy) return;
    const center = { x: cy.width() / 2, y: cy.height() / 2 };
    const newZoom = Math.max(cy.minZoom(), Math.min(cy.maxZoom(), cy.zoom() * factor));
    cy.zoom({ level: newZoom, renderedPosition: center });
  }

  function fit() {
    if (!cy || !cy.elements().length) return;
    cy.fit(undefined, 40);
  }

  function resize() {
    if (!cy) return;
    cy.resize();
  }

  window.KGraph = { init, merge, clear, removeNode, applyMergeGroups, exportJSON, size, onNodeClick, zoomBy, fit, resize };
})();
