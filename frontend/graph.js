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
  const NODE_TYPE_COLORS = {
    CHEMICAL:             'var(--node-chemical)',
    DISEASE:              'var(--node-disease)',
    GENE_OR_GENE_PRODUCT: 'var(--node-gene)',
    ANATOMY:              'var(--node-anatomy)',
    SYMPTOM:              'var(--node-symptom)',
    PROCEDURE:            'var(--node-procedure)',
    CELL_TYPE:            'var(--node-anatomy)',
    ORGANISM:             'var(--node-default)',
    OTHER:                'var(--node-default)',
  };

  const state = {
    nodes: new Map(),   // id -> { id, label, type, citations: [], kb_id }
    edges: new Map(),   // id -> { id, source, target, predicate, citations: [] }
  };

  let cy = null;
  let nodeClickHandlers = [];

  function init(container) {
    if (typeof cytoscape === 'undefined') {
      console.error('[KGraph] cytoscape library not loaded');
      return null;
    }
    if (typeof cytoscape.use === 'function' && typeof window.cytoscapeFcose !== 'undefined') {
      try { cytoscape.use(window.cytoscapeFcose); } catch (e) { /* already registered */ }
    }

    cy = cytoscape({
      container: container,
      elements: [],
      wheelSensitivity: 0.25,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele) => NODE_TYPE_COLORS[ele.data('type')] || NODE_TYPE_COLORS.OTHER,
            'label':            'data(label)',
            'color':            '#F1F5F9',
            'font-size':        '11px',
            'font-weight':      '600',
            'text-valign':      'bottom',
            'text-halign':      'center',
            'text-margin-y':    6,
            'text-outline-color': '#0A0E1A',
            'text-outline-width': 2,
            'border-width':     2,
            'border-color':     'rgba(255,255,255,.18)',
            'width':            32,
            'height':           32,
            'transition-property': 'border-color, border-width, background-color, width, height',
            'transition-duration': '180ms',
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#FFFFFF',
            'border-width': 3,
            'width':  40,
            'height': 40,
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
            'width':             1.5,
            'line-color':        'rgba(255,255,255,.18)',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': 'rgba(255,255,255,.30)',
            'arrow-scale':       0.9,
            'label':             'data(predicate)',
            'font-size':         '9px',
            'color':             'rgba(255,255,255,.75)',
            'text-rotation':     'autorotate',
            'text-background-color': '#0A0E1A',
            'text-background-opacity': 0.85,
            'text-background-padding': '2px',
            'text-background-shape':   'roundrectangle',
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
    const useFcose = cy.elements().length > 0 && cy.extension && cy.extension('layout', 'fcose');
    cy.layout({
      name: useFcose ? 'fcose' : 'cose',
      animate: 'end',
      animationDuration: 350,
      randomize: false,
      nodeRepulsion: 6500,
      idealEdgeLength: 90,
      edgeElasticity: 0.45,
      nestingFactor: 1.2,
      gravity: 0.25,
      fit: true,
      padding: 30,
    }).run();
  }

  function merge(payload) {
    if (!cy || !payload) return { addedNodes: [], addedEdges: [] };

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

  window.KGraph = { init, merge, clear, exportJSON, size, onNodeClick };
})();
