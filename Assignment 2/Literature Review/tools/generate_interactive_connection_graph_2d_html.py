"""
Generate a self-contained interactive 2D connection graph HTML (no Three.js).

Why 2D:
- Cleaner aesthetics than 3D spheres
- No Three.js dependency warnings
- Stable filtering (we keep immutable source_id/target_id)

Uses `force-graph` (Canvas) via CDN so the HTML opens directly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalise_edge_type(value: Any) -> str:
    if value is None:
        return "(missing)"
    if isinstance(value, list):
        parts = [str(v).strip().lower() for v in value if str(v).strip()]
        parts = sorted(set(parts))
        return "+".join(parts) if parts else "(missing)"
    s = str(value).strip().lower()
    return s if s else "(missing)"


def _normalise_title(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(str(value).strip())
    except Exception:
        return None


def _sort_paper_ids(ids: List[str]) -> List[str]:
    return sorted(ids, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))


def build_graph_data(raw: Dict[str, Any]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = raw.get("nodes", [])
    edges: List[Dict[str, Any]] = raw.get("edges", [])

    raw_nodes: List[Dict[str, Any]] = []
    for n in nodes:
        node_id = str(n.get("id", "")).strip()
        if not node_id:
            continue
        raw_nodes.append(
            {
                "id": node_id,
                "paper_id": str(n.get("paper_id", node_id)).strip(),
                "title": str(n.get("title", "")).strip(),
                "bucket": str(n.get("bucket", "")).strip(),
                "tier": str(n.get("tier", "")).strip(),
                "year": str(n.get("year", "")).strip(),
                "priority_score": n.get("priority_score", None),
            }
        )

    # Collapse duplicates (e.g. paper IDs 2 and 4 are the same paper).
    # Canonical key: (normalised title, year int if parseable).
    by_key: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {}
    for n in raw_nodes:
        key = (_normalise_title(n.get("title")), _safe_int(n.get("year")))
        if not key[0]:
            # If title missing, treat as unique by id.
            key = (f"__id__{n['id']}", key[1])
        by_key.setdefault(key, []).append(n)

    canonical_of: Dict[str, str] = {}
    graph_nodes: List[Dict[str, Any]] = []
    for _key, group in by_key.items():
        ids = _sort_paper_ids([g["id"] for g in group])
        canonical_id = ids[0]
        for g in group:
            canonical_of[g["id"]] = canonical_id

        # Pick canonical node payload (prefer highest priority_score if present)
        canonical_node = None
        best_score = -1.0
        for g in group:
            score = g.get("priority_score")
            try:
                score_f = float(score) if score is not None else -1.0
            except Exception:
                score_f = -1.0
            if g["id"] == canonical_id:
                # small bonus to keep canonical stable
                score_f += 0.001
            if score_f > best_score:
                best_score = score_f
                canonical_node = g

        assert canonical_node is not None
        aliases = [i for i in ids[1:]]
        out = dict(canonical_node)
        out["aliases"] = aliases
        graph_nodes.append(out)

    # Remap + de-duplicate links after collapsing.
    # Key by (src, tgt, type) undirected (so src/tgt order doesn't matter).
    collapsed_links: Dict[Tuple[str, str, str], float] = {}
    for e in edges:
        src_raw = str(e.get("source", "")).strip()
        tgt_raw = str(e.get("target", "")).strip()
        if not src_raw or not tgt_raw:
            continue
        src = canonical_of.get(src_raw, src_raw)
        tgt = canonical_of.get(tgt_raw, tgt_raw)
        if not src or not tgt or src == tgt:
            continue
        edge_type = _normalise_edge_type(e.get("type", None) if "type" in e else e.get("connection_type", None))
        w = e.get("weight", 1.0)
        try:
            w_f = float(w) if w is not None else 1.0
        except Exception:
            w_f = 1.0
        a, b = (src, tgt) if src <= tgt else (tgt, src)
        key = (a, b, edge_type)
        collapsed_links[key] = max(collapsed_links.get(key, 0.0), w_f)

    graph_links: List[Dict[str, Any]] = []
    for (src, tgt, edge_type), w_f in sorted(collapsed_links.items(), key=lambda kv: (-kv[1], kv[0][2], kv[0][0], kv[0][1])):
        graph_links.append(
            {
                "source": src,
                "target": tgt,
                "source_id": src,
                "target_id": tgt,
                "type": edge_type,
                "weight": w_f,
            }
        )

    return {"nodes": graph_nodes, "links": graph_links}


def render_html(graph_data: Dict[str, Any], title: str) -> str:
    graph_data_json = json.dumps(graph_data, ensure_ascii=False)
    build_id = str(__import__("datetime").datetime.now().strftime("%Y-%m-%d__%H%M%S"))

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      html, body {{
        margin: 0;
        height: 100%;
        width: 100%;
        overflow: hidden;
        background: #0b0e14;
        color: #e6e6e6;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      }}

      #graph {{
        position: absolute;
        inset: 0 0 0 380px;
        z-index: 0;
      }}

      #panel {{
        position: absolute;
        top: 12px;
        left: 12px;
        bottom: 12px;
        width: 352px;
        padding: 12px 12px;
        background: rgba(0, 0, 0, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        backdrop-filter: blur(8px);
        overflow: auto;
        z-index: 10;
      }}

      #graph canvas {{
        pointer-events: auto;
        touch-action: none;
      }}

      #panel h1 {{
        font-size: 14px;
        margin: 0 0 8px 0;
        font-weight: 700;
      }}

      .row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        font-size: 12px;
        opacity: 0.95;
      }}

      .pill {{
        display: inline-block;
        padding: 2px 8px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 999px;
        font-size: 11px;
        opacity: 0.9;
      }}

      .section {{
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
      }}

      .section-title {{
        font-size: 12px;
        font-weight: 750;
        margin: 0 0 8px 0;
        opacity: 0.95;
      }}

      .control {{
        font-size: 12px;
        margin: 8px 0;
      }}

      .control label {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        user-select: none;
      }}

      .control input[type="text"] {{
        width: 100%;
        box-sizing: border-box;
        padding: 10px 10px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.06);
        color: #e6e6e6;
        outline: none;
      }}

      .control input[type="range"] {{
        width: 100%;
      }}

      .grid2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px 10px;
      }}

      .btn {{
        appearance: none;
        border: 1px solid rgba(255, 255, 255, 0.14);
        background: rgba(255, 255, 255, 0.06);
        color: #e6e6e6;
        padding: 8px 10px;
        border-radius: 12px;
        cursor: pointer;
        font-size: 12px;
      }}

      .btn:hover {{
        background: rgba(255, 255, 255, 0.10);
      }}

      .small {{
        font-size: 12px;
        opacity: 0.85;
      }}

      #details {{
        font-size: 12px;
        white-space: pre-wrap;
        line-height: 1.35;
        opacity: 0.95;
      }}

      #neighbours {{
        margin-top: 8px;
        font-size: 12px;
        opacity: 0.9;
      }}

      #neighbours a {{
        color: #93c5fd;
        text-decoration: none;
      }}

      #neighbours a:hover {{
        text-decoration: underline;
      }}

      #nodePicker a {{
        color: #93c5fd;
        text-decoration: none;
      }}

      #nodePicker a:hover {{
        text-decoration: underline;
      }}

      #legend {{
        margin-top: 10px;
        display: grid;
        grid-template-columns: 12px 1fr;
        gap: 6px 10px;
        font-size: 12px;
        opacity: 0.9;
      }}

      .swatch {{
        width: 12px;
        height: 12px;
        border-radius: 3px;
      }}

      @media (max-width: 980px) {{
        #graph {{ inset: 0; }}
        #panel {{
          left: 12px;
          right: 12px;
          width: auto;
          max-height: 45vh;
        }}
      }}

      body.panel-collapsed #panel {{
        display: none;
      }}

      body.panel-collapsed #graph {{
        inset: 0;
      }}
    </style>
  </head>
  <body>
    <div id="graph"></div>

    <div id="panel">
      <h1>{title}</h1>
      <div class="row small">
        <div class="pill" id="statNodes" role="status" aria-live="polite" aria-label="nodes: ?">nodes: ?</div>
        <div class="pill" id="statEdges" role="status" aria-live="polite" aria-label="edges: ?">edges: ?</div>
        <div class="pill" id="statYears" role="status" aria-live="polite" aria-label="years: ?–?">years: ?–?</div>
        <div class="pill" id="statBuild" role="status" aria-live="polite" aria-label="build: {build_id}">build: {build_id}</div>
        <div class="pill" id="statEvent" role="status" aria-live="polite" aria-label="event: (load)">event: (load)</div>
      </div>

      <div class="section">
        <div class="section-title">Search</div>
        <div class="control">
          <input id="searchBox" type="text" placeholder="paper id or title (press Enter)" />
        </div>
        <div class="control grid2">
          <button class="btn" id="btnClearSearch">Clear</button>
          <button class="btn" id="btnResetView">Reset view</button>
          <button class="btn" id="btnFitGraph">Fit graph</button>
          <button class="btn" id="btnTogglePanel">Toggle panel</button>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Filters</div>

        <div class="control">
          <label><input type="checkbox" id="toggleLabels" /> Show labels</label>
        </div>

        <div class="control">
          <label><input type="checkbox" id="toggleTitles" /> Label titles (instead of IDs)</label>
        </div>

        <div class="control">
          <label><input type="checkbox" id="toggleFocus" /> Focus mode (selected + neighbours)</label>
        </div>

        <div class="control">
          <label><input type="checkbox" id="toggleDebugPointer" /> Debug pointer events</label>
          <div class="small" id="pointerDebugText" style="margin-top:6px;">pointer: (off)</div>
          <button class="btn" id="btnTestCanvasPointer" style="margin-top:8px;">Test canvas pointer</button>
        </div>

        <div class="control">
          <div class="small"><strong>Year range</strong>: <span id="yearRangeText">?</span></div>
          <input id="yearMin" type="range" min="2000" max="2030" value="2000" />
          <input id="yearMax" type="range" min="2000" max="2030" value="2030" />
        </div>

        <div class="control">
          <div class="small"><strong>Edge types</strong></div>
          <div id="edgeTypeChecks"></div>
        </div>

        <div class="control">
          <div class="small"><strong>Buckets</strong></div>
          <div id="bucketChecks"></div>
        </div>

        <div class="control">
          <div class="small"><strong>Tiers</strong></div>
          <div id="tierChecks"></div>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Selection</div>
        <div class="control grid2">
          <input id="gotoNodeId" type="text" placeholder="node id (e.g. 2)" />
          <button class="btn" id="btnGotoNode">Go</button>
        </div>
        <div class="small" style="margin-top:8px;"><strong>Node picker</strong> (filtered, clickable)</div>
        <div id="nodePicker" class="small" style="margin-top:6px; display:grid; gap:6px;"></div>
        <div id="details" class="small">Click a node to see details.</div>
        <div id="neighbours"></div>
      </div>

      <div class="section">
        <div class="section-title">Legend</div>
        <div id="legend">
          <div class="swatch" style="background:#38bdf8"></div><div>thematic</div>
          <div class="swatch" style="background:#22c55e"></div><div>citation</div>
          <div class="swatch" style="background:#f59e0b"></div><div>semantic</div>
          <div class="swatch" style="background:#a78bfa"></div><div>multi-type (e.g. citation+thematic)</div>
          <div class="swatch" style="background:#94a3b8"></div><div>(missing)</div>
        </div>
        <div class="small" style="margin-top:8px;">Controls: drag • scroll zoom • click selects</div>
      </div>
    </div>

    <script src="https://unpkg.com/force-graph@1.51.0/dist/force-graph.min.js"></script>
    <script>
      const graphData = {graph_data_json};
      const BUILD_ID = "{build_id}";

      const EDGE_COLOURS = {{
        thematic: '#38bdf8',
        citation: '#22c55e',
        semantic: '#f59e0b',
        '(missing)': '#94a3b8'
      }};

      function hexToRgba(hex, alpha) {{
        const h = String(hex || '').replace('#','').trim();
        if (h.length !== 6) return `rgba(148,163,184,${{alpha}})`;
        const r = parseInt(h.slice(0,2), 16);
        const g = parseInt(h.slice(2,4), 16);
        const b = parseInt(h.slice(4,6), 16);
        return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
      }}

      function edgeColour(edgeType) {{
        const t = String(edgeType || '(missing)').trim().toLowerCase();
        if (EDGE_COLOURS[t]) return EDGE_COLOURS[t];
        if (t.includes('+')) return '#a78bfa';
        return '#94a3b8';
      }}

      function nodeFill(bucket) {{
        const b = String(bucket || '').toLowerCase();
        if (b.includes('bucket_a')) return '#ef4444';
        if (b.includes('bucket_b')) return '#f97316';
        if (b.includes('bucket_c')) return '#22c55e';
        if (b.includes('bucket_d')) return '#38bdf8';
        if (b.includes('clinical')) return '#a78bfa';
        if (b.includes('evaluation')) return '#eab308';
        return '#94a3b8';
      }}

      const baseNodes = graphData.nodes.map(n => ({{
        ...n,
        year_num: Number(String(n.year || '').replace(/\\D+/g,'')) || null
      }}));

      const baseLinks = graphData.links.map(l => ({{
        source_id: String(l.source_id || l.source || '').trim(),
        target_id: String(l.target_id || l.target || '').trim(),
        type: String(l.type || '(missing)').trim().toLowerCase(),
        weight: (l.weight === null || l.weight === undefined) ? 1.0 : l.weight
      }}));

      const allBuckets = Array.from(new Set(baseNodes.map(n => n.bucket).filter(Boolean))).sort();
      const allTiers = Array.from(new Set(baseNodes.map(n => n.tier).filter(Boolean))).sort();
      const allEdgeTypes = Array.from(new Set(baseLinks.map(l => l.type).filter(Boolean))).sort();

      const years = baseNodes.map(n => n.year_num).filter(y => Number.isFinite(y));
      const yearMinAll = years.length ? Math.min(...years) : 2000;
      const yearMaxAll = years.length ? Math.max(...years) : 2030;

      const state = {{
        search: '',
        selectedNodeId: null,
        showLabels: false,
        labelTitles: false,
        focusMode: false,
        yearMin: yearMinAll,
        yearMax: yearMaxAll,
        buckets: new Set(allBuckets),
        tiers: new Set(allTiers),
        edgeTypes: new Set(allEdgeTypes)
      }};

      let hoveredNodeId = null;
      let currentAdj = new Map(); // rebuilt on refresh()

      const elStatNodes = document.getElementById('statNodes');
      const elStatEdges = document.getElementById('statEdges');
      const elStatYears = document.getElementById('statYears');
      const elStatBuild = document.getElementById('statBuild');
      const elStatEvent = document.getElementById('statEvent');

      function setEvent(message) {{
        if (!elStatEvent) return;
        elStatEvent.textContent = String(message || '');
        elStatEvent.setAttribute('aria-label', elStatEvent.textContent);
      }}

      function setPointerDebug(message) {{
        if (!elPointerDebugText) return;
        elPointerDebugText.textContent = String(message || '');
      }}
      const elYearRangeText = document.getElementById('yearRangeText');
      const elYearMin = document.getElementById('yearMin');
      const elYearMax = document.getElementById('yearMax');
      const elEdgeTypeChecks = document.getElementById('edgeTypeChecks');
      const elBucketChecks = document.getElementById('bucketChecks');
      const elTierChecks = document.getElementById('tierChecks');
      const elSearchBox = document.getElementById('searchBox');
      const elToggleLabels = document.getElementById('toggleLabels');
      const elToggleTitles = document.getElementById('toggleTitles');
      const elToggleFocus = document.getElementById('toggleFocus');
      const elToggleDebugPointer = document.getElementById('toggleDebugPointer');
      const elPointerDebugText = document.getElementById('pointerDebugText');
      const elBtnTestCanvasPointer = document.getElementById('btnTestCanvasPointer');
      const elGotoNodeId = document.getElementById('gotoNodeId');
      const elBtnGotoNode = document.getElementById('btnGotoNode');
      const elNodePicker = document.getElementById('nodePicker');
      const elDetails = document.getElementById('details');
      const elNeighbours = document.getElementById('neighbours');

      elYearMin.min = String(yearMinAll);
      elYearMin.max = String(yearMaxAll);
      elYearMax.min = String(yearMinAll);
      elYearMax.max = String(yearMaxAll);
      elYearMin.value = String(yearMinAll);
      elYearMax.value = String(yearMaxAll);

      function clampRangePair(minEl, maxEl) {{
        const a = Number(minEl.value);
        const b = Number(maxEl.value);
        if (a <= b) return [a, b];
        return [b, a];
      }}

      function mkCheckRow(container, id, label, checked, onChange) {{
        const div = document.createElement('div');
        div.className = 'control';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.id = id;
        cb.checked = checked;
        cb.addEventListener('change', () => onChange(cb.checked));
        const lab = document.createElement('label');
        lab.htmlFor = id;
        lab.appendChild(cb);
        const span = document.createElement('span');
        span.textContent = label;
        lab.appendChild(span);
        div.appendChild(lab);
        container.appendChild(div);
      }}

      function renderChecklists() {{
        elEdgeTypeChecks.innerHTML = '';
        allEdgeTypes.forEach(t => {{
          const id = `edgeType_${{t.replace(/[^a-z0-9_+]/g,'_')}}`;
          mkCheckRow(elEdgeTypeChecks, id, t, state.edgeTypes.has(t), (isOn) => {{
            if (isOn) state.edgeTypes.add(t); else state.edgeTypes.delete(t);
            setEvent('event: edge type ' + t + ' -> ' + (isOn ? 'on' : 'off'));
            refresh();
          }});
        }});

        elBucketChecks.innerHTML = '';
        allBuckets.forEach(b => {{
          const id = `bucket_${{b.replace(/[^a-z0-9_]/g,'_')}}`;
          mkCheckRow(elBucketChecks, id, b, state.buckets.has(b), (isOn) => {{
            if (isOn) state.buckets.add(b); else state.buckets.delete(b);
            setEvent('event: bucket ' + b + ' -> ' + (isOn ? 'on' : 'off'));
            refresh();
          }});
        }});

        elTierChecks.innerHTML = '';
        allTiers.forEach(t => {{
          const id = `tier_${{t.replace(/[^a-z0-9_]/g,'_')}}`;
          mkCheckRow(elTierChecks, id, t, state.tiers.has(t), (isOn) => {{
            if (isOn) state.tiers.add(t); else state.tiers.delete(t);
            setEvent('event: tier ' + t + ' -> ' + (isOn ? 'on' : 'off'));
            refresh();
          }});
        }});
      }}

      function formatNodeDetails(n) {{
        if (!n) return 'Click a node to see details.';
        const parts = [];
        parts.push(`ID: ${{n.id}}`);
        if (n.aliases && n.aliases.length) parts.push(`Also listed as: ${{n.aliases.join(', ')}}`);
        if (n.title) parts.push(`Title: ${{n.title}}`);
        if (n.bucket) parts.push(`Bucket: ${{n.bucket}}`);
        if (n.tier) parts.push(`Tier: ${{n.tier}}`);
        if (n.year) parts.push(`Year: ${{n.year}}`);
        if (n.priority_score !== null && n.priority_score !== undefined) parts.push(`Priority: ${{n.priority_score}}`);
        return parts.join('\\n');
      }}

      function rebuildAdjacency(linksForGraph) {{
        const adj = new Map();
        linksForGraph.forEach(l => {{
          const s = String(l.source_id || l.source || '').trim();
          const t = String(l.target_id || l.target || '').trim();
          if (!s || !t) return;
          if (!adj.has(s)) adj.set(s, new Set());
          if (!adj.has(t)) adj.set(t, new Set());
          adj.get(s).add(t);
          adj.get(t).add(s);
        }});
        currentAdj = adj;
      }}

      function computeDegreeById(linksForGraph) {{
        const degree = new Map();
        (linksForGraph || []).forEach(l => {{
          const s = String(l.source_id || l.source || '').trim();
          const t = String(l.target_id || l.target || '').trim();
          if (!s || !t) return;
          degree.set(s, (degree.get(s) || 0) + 1);
          degree.set(t, (degree.get(t) || 0) + 1);
        }});
        return degree;
      }}

      function shortTitle(n) {{
        const raw = String((n && n.title) ? n.title : '').trim();
        if (!raw) return '';
        return raw.length > 58 ? (raw.slice(0, 57) + '…') : raw;
      }}

      function renderNeighboursForSelected(nodeId) {{
        const id = String(nodeId || '').trim();
        const current = Graph.graphData();
        const neighbours = [];
        if (id && current.links) {{
          current.links.forEach(l => {{
            const srcId = (typeof l.source === 'object' && l.source) ? l.source.id : l.source;
            const tgtId = (typeof l.target === 'object' && l.target) ? l.target.id : l.target;
            if (String(srcId) === id) neighbours.push(tgtId);
            else if (String(tgtId) === id) neighbours.push(srcId);
          }});
        }}

        const uniq = Array.from(new Set(neighbours)).sort((a, b) => Number(a) - Number(b));
        if (!uniq.length) {{
          elNeighbours.innerHTML = '';
          return;
        }}

        const nodeById = new Map((Graph.graphData().nodes || []).map(n => [String(n.id), n]));
        const fmt = (nid) => {{
          const n = nodeById.get(String(nid));
          if (!n) return String(nid);
          const t = shortTitle(n);
          return t ? (String(n.id) + ': ' + t) : String(n.id);
        }};

        elNeighbours.innerHTML =
          '<div class="small"><strong>Neighbours</strong> (click to focus):</div>' +
          '<div class="small">' +
          uniq.map(nid => '<a href="#" data-id="' + String(nid) + '">' + fmt(nid) + '</a>').join('<br/>') +
          '</div>';

        elNeighbours.querySelectorAll('a[data-id]').forEach(a => {{
          a.addEventListener('click', (ev) => {{
            ev.preventDefault();
            const nid = ev.currentTarget.getAttribute('data-id');
            if (!nid) return;
            selectNodeById(nid);
          }});
        }});
      }}

      function selectNodeById(nodeId) {{
        const id = String(nodeId || '').trim();
        if (!id) return;
        const n = (Graph.graphData().nodes || []).find(nn => String(nn.id) === id);
        if (!n) {{
          setEvent('event: select node ' + id + ' (not in current filter)');
          return;
        }}

        state.selectedNodeId = n.id;
        elDetails.textContent = formatNodeDetails(n);
        setEvent('event: select node ' + id);
        renderNeighboursForSelected(n.id);

        try {{
          Graph.centerAt(n.x, n.y, 600);
          Graph.zoom(3, 600);
        }} catch (e) {{}}

        refresh();
      }}

      function renderNodePicker(filteredNodes, linksForGraph) {{
        if (!elNodePicker) return;
        const degree = computeDegreeById(linksForGraph);

        const nodesSorted = (filteredNodes || []).slice().sort((a, b) => {{
          const da = degree.get(String(a.id)) || 0;
          const db = degree.get(String(b.id)) || 0;
          if (db !== da) return db - da;
          return String(a.id).localeCompare(String(b.id), undefined, {{ numeric: true }});
        }});

        const MAX_ITEMS = 18;
        const subset = nodesSorted.slice(0, MAX_ITEMS);

        const rows = subset.map(n => {{
          const id = String(n.id);
          const title = shortTitle(n);
          const label = title ? (id + ': ' + title) : id;
          return '<a href="#" data-node-id="' + id + '">' + label + '</a>';
        }});

        if (nodesSorted.length > MAX_ITEMS) {{
          rows.push('<div style="opacity:0.75; margin-top:6px;">…and ' + String(nodesSorted.length - MAX_ITEMS) + ' more (use search)</div>');
        }}

        elNodePicker.innerHTML = rows.join('<br/>');
        elNodePicker.querySelectorAll('a[data-node-id]').forEach(a => {{
          a.addEventListener('click', (ev) => {{
            ev.preventDefault();
            const nid = ev.currentTarget.getAttribute('data-node-id');
            if (!nid) return;
            selectNodeById(nid);
          }});
        }});
      }}

      function computeFiltered() {{
        const yMin = Math.min(state.yearMin, state.yearMax);
        const yMax = Math.max(state.yearMin, state.yearMax);
        const search = state.search.trim().toLowerCase();

        let nodes = baseNodes.filter(n => {{
          if (n.bucket && !state.buckets.has(n.bucket)) return false;
          if (n.tier && !state.tiers.has(n.tier)) return false;
          if (Number.isFinite(n.year_num)) {{
            if (n.year_num < yMin || n.year_num > yMax) return false;
          }}
          if (search) {{
            const hay = `${{n.id}} ${{n.title || ''}}`.toLowerCase();
            if (!hay.includes(search)) return false;
          }}
          return true;
        }});

        const nodeIdSet = new Set(nodes.map(n => n.id));
        let links = baseLinks.filter(l => {{
          if (!nodeIdSet.has(l.source_id) || !nodeIdSet.has(l.target_id)) return false;
          if (!state.edgeTypes.has(l.type)) return false;
          return true;
        }});

        if (state.focusMode && state.selectedNodeId && nodeIdSet.has(state.selectedNodeId)) {{
          const keep = new Set([state.selectedNodeId]);
          links.forEach(l => {{
            if (l.source_id === state.selectedNodeId) keep.add(l.target_id);
            if (l.target_id === state.selectedNodeId) keep.add(l.source_id);
          }});
          nodes = nodes.filter(n => keep.has(n.id));
          const keepIds = new Set(nodes.map(n => n.id));
          links = links.filter(l => keepIds.has(l.source_id) && keepIds.has(l.target_id));
        }}

        return {{ nodes, links }};
      }}

      const Graph = ForceGraph()(document.getElementById('graph'))
        .backgroundColor('#0b0e14')
        .nodeId('id')
        .nodeLabel(n => {{
          const parts = [
            `ID: ${{n.id}}`,
            n.title ? `Title: ${{n.title}}` : null,
            n.bucket ? `Bucket: ${{n.bucket}}` : null,
            n.tier ? `Tier: ${{n.tier}}` : null,
            n.year ? `Year: ${{n.year}}` : null
          ].filter(Boolean);
          return parts.join('\\n');
        }})
        .linkColor(l => hexToRgba(edgeColour(l.type), 0.22))
        .linkWidth(l => {{
          const w = Number(l.weight);
          if (Number.isFinite(w)) return 0.7 + Math.min(3.0, w * 0.18);
          return 0.9;
        }})
        .nodeRelSize(4)
        .nodeCanvasObject((node, ctx, globalScale) => {{
          const nodeId = String(node.id);
          const isSelected = state.selectedNodeId && nodeId === String(state.selectedNodeId);
          const isHovered = hoveredNodeId && nodeId === String(hoveredNodeId);
          const neighbourSet = hoveredNodeId ? (currentAdj.get(String(hoveredNodeId)) || new Set()) : new Set();
          const isNeighbour = hoveredNodeId ? neighbourSet.has(nodeId) : false;
          const dim = hoveredNodeId && !isHovered && !isNeighbour;
          const alpha = dim ? 0.14 : 1.0;

          const priority = Number(node.priority_score);
          const priorityScore = Number.isFinite(priority) ? Math.min(6, Math.max(0, priority)) : 0;
          const rBase = 3.2 + (priorityScore * 0.55); // smaller circles
          const r = (isSelected || isHovered) ? rBase + 2.0 : rBase;

          ctx.beginPath();
          ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
          ctx.fillStyle = hexToRgba(nodeFill(node.bucket), alpha);
          ctx.fill();
          ctx.lineWidth = (isSelected || isHovered) ? 2.8 : 1.4;
          ctx.strokeStyle = (isSelected || isHovered) ? 'rgba(255,255,255,0.95)' : 'rgba(0,0,0,0.55)';
          ctx.stroke();

          if (!state.showLabels && !isSelected && !isHovered) return;

          let label = nodeId;
          if (state.labelTitles && node.title) label = String(node.title);
          if (label.length > 44) label = label.slice(0, 43) + '…';

          const fontSize = Math.max(9, 13 / globalScale);
          ctx.font = `${{fontSize}}px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = 'rgba(0,0,0,0.75)';
          ctx.fillText(label, node.x + 1, node.y - r - fontSize - 1);
          ctx.fillStyle = 'rgba(230,230,230,0.95)';
          ctx.fillText(label, node.x, node.y - r - fontSize);
        }})
        // Critical: when using custom nodeCanvasObject, we must also paint pointer areas
        // or the library can't reliably hit-test nodes for hover/click.
        .nodePointerAreaPaint((node, colour, ctx) => {{
          const priority = Number(node.priority_score);
          const priorityScore = Number.isFinite(priority) ? Math.min(6, Math.max(0, priority)) : 0;
          const rBase = 3.2 + (priorityScore * 0.55);
          const rHit = rBase + 8.0; // larger hit zone for finicky touchpads
          ctx.fillStyle = colour;
          ctx.beginPath();
          ctx.arc(node.x, node.y, rHit, 0, 2 * Math.PI, false);
          ctx.fill();
        }})
        .onNodeHover(node => {{
          hoveredNodeId = node ? node.id : null;
          Graph.refresh();
        }})
        .onNodeClick(node => {{
          state.selectedNodeId = node ? node.id : null;
          elDetails.textContent = formatNodeDetails(node);
          setEvent(node ? ('event: node click ' + node.id) : 'event: node click (null)');
          renderNeighboursForSelected(node ? node.id : null);

          if (node) {{
            Graph.centerAt(node.x, node.y, 600);
            Graph.zoom(3, 600);
          }}
          refresh();
        }});

      function refresh() {{
        if (elStatBuild) elStatBuild.textContent = `build: ${{BUILD_ID}}`;
        if (elStatBuild) elStatBuild.setAttribute('aria-label', elStatBuild.textContent);
        const [a, b] = clampRangePair(elYearMin, elYearMax);
        // Keep slider UI ordered (if crossed, clamp max to min)
        if (Number(elYearMin.value) > Number(elYearMax.value)) {{
          elYearMax.value = String(elYearMin.value);
        }}
        state.yearMin = Number(elYearMin.value);
        state.yearMax = Number(elYearMax.value);

        const yMin = Math.min(state.yearMin, state.yearMax);
        const yMax = Math.max(state.yearMin, state.yearMax);
        elYearRangeText.textContent = `${{yMin}}–${{yMax}}`;

        const filtered = computeFiltered();
        const linksForGraph = filtered.links.map(l => ({{
          source: l.source_id,
          target: l.target_id,
          type: l.type,
          weight: l.weight
        }}));

        Graph.graphData({{ nodes: filtered.nodes, links: linksForGraph }});
        rebuildAdjacency(linksForGraph);
        renderNodePicker(filtered.nodes, linksForGraph);

        elStatNodes.textContent = `nodes: ${{filtered.nodes.length}}`;
        elStatEdges.textContent = `edges: ${{linksForGraph.length}}`;
        elStatYears.textContent = `years: ${{yMin}}–${{yMax}}`;
        elStatNodes.setAttribute('aria-label', elStatNodes.textContent);
        elStatEdges.setAttribute('aria-label', elStatEdges.textContent);
        elStatYears.setAttribute('aria-label', elStatYears.textContent);
      }}

      renderChecklists();
      refresh();

      // Nicer layout tuning (reduce clumping; safe if the underlying lib exposes these)
      try {{
        Graph.d3Force('charge').strength(-220);
        Graph.d3Force('link').distance(70);
        Graph.d3VelocityDecay(0.30);
      }} catch (e) {{}}

      // Explicitly enable pointer interaction + node dragging (some environments disable by default).
      try {{
        Graph.enablePointerInteraction(true);
        Graph.enableNodeDrag(true);
      }} catch (e) {{}}

      // Drag event diagnostics (shows whether canvas->graph interaction is working).
      try {{
        Graph.onNodeDrag((node) => {{
          if (node && node.id !== undefined) setEvent('event: drag ' + node.id);
        }});
        Graph.onNodeDragEnd((node) => {{
          if (node && node.id !== undefined) setEvent('event: drag end ' + node.id);
        }});
      }} catch (e) {{}}

      // Canvas-level pointer diagnostics: if these never fire, the canvas isn't receiving pointer events.
      setTimeout(() => {{
        const canvas = document.querySelector('#graph canvas');
        if (!canvas) {{
          setEvent('event: canvas not found');
          return;
        }}
        canvas.style.cursor = 'grab';
        canvas.addEventListener('pointerdown', (ev) => {{
          canvas.style.cursor = 'grabbing';
          setEvent('event: canvas pointerdown');
          if (elToggleDebugPointer && elToggleDebugPointer.checked) {{
            setPointerDebug('pointer: down target=canvas button=' + String(ev.button) + ' x=' + String(ev.clientX) + ' y=' + String(ev.clientY));
          }}
        }});
        canvas.addEventListener('pointerup', (ev) => {{
          canvas.style.cursor = 'grab';
          setEvent('event: canvas pointerup');
          if (elToggleDebugPointer && elToggleDebugPointer.checked) {{
            setPointerDebug('pointer: up target=canvas button=' + String(ev.button) + ' x=' + String(ev.clientX) + ' y=' + String(ev.clientY));
          }}
        }});
      }}, 0);

      elYearMin.addEventListener('input', refresh);
      elYearMax.addEventListener('input', refresh);

      elToggleLabels.addEventListener('change', () => {{
        state.showLabels = !!elToggleLabels.checked;
        setEvent('event: toggle labels');
        refresh();
      }});

      elToggleTitles.addEventListener('change', () => {{
        state.labelTitles = !!elToggleTitles.checked;
        setEvent('event: toggle titles');
        refresh();
      }});

      elToggleFocus.addEventListener('change', () => {{
        state.focusMode = !!elToggleFocus.checked;
        setEvent('event: toggle focus');
        refresh();
      }});

      if (elToggleDebugPointer) {{
        elToggleDebugPointer.addEventListener('change', () => {{
          setPointerDebug(elToggleDebugPointer.checked ? 'pointer: (armed) click/drag on graph' : 'pointer: (off)');
        }});
      }}

      elSearchBox.addEventListener('keydown', (ev) => {{
        if (ev.key !== 'Enter') return;
        state.search = elSearchBox.value || '';
        setEvent('event: search "' + state.search + '"');
        refresh();
      }});

      if (elBtnTestCanvasPointer) {{
        elBtnTestCanvasPointer.addEventListener('click', () => {{
          const canvas = document.querySelector('#graph canvas');
          if (!canvas) {{
            setEvent('event: test pointer (no canvas)');
            return;
          }}
          const rect = canvas.getBoundingClientRect();
          const x = rect.left + rect.width * 0.5;
          const y = rect.top + rect.height * 0.5;
          setEvent('event: test pointer dispatch');
          try {{
            canvas.dispatchEvent(new PointerEvent('pointerdown', {{ bubbles: true, clientX: x, clientY: y, pointerId: 1, pointerType: 'mouse' }}));
            canvas.dispatchEvent(new PointerEvent('pointerup', {{ bubbles: true, clientX: x, clientY: y, pointerId: 1, pointerType: 'mouse' }}));
            if (elToggleDebugPointer && elToggleDebugPointer.checked) {{
              setPointerDebug('pointer: dispatched to canvas at x=' + String(Math.round(x)) + ' y=' + String(Math.round(y)));
            }}
          }} catch (e) {{
            setEvent('event: test pointer failed');
          }}
        }});
      }}

      elGotoNodeId.addEventListener('keydown', (ev) => {{
        if (ev.key !== 'Enter') return;
        selectNodeById(elGotoNodeId.value || '');
      }});

      elBtnGotoNode.addEventListener('click', () => {{
        selectNodeById(elGotoNodeId.value || '');
      }});

      document.getElementById('btnClearSearch').addEventListener('click', () => {{
        elSearchBox.value = '';
        state.search = '';
        setEvent('event: clear search');
        refresh();
      }});

      document.getElementById('btnFitGraph').addEventListener('click', () => {{
        setEvent('event: fit graph');
        try {{ Graph.zoomToFit(700, 40); }} catch (e) {{}}
      }});

      document.getElementById('btnTogglePanel').addEventListener('click', () => {{
        document.body.classList.toggle('panel-collapsed');
        setEvent('event: panel ' + (document.body.classList.contains('panel-collapsed') ? 'collapsed' : 'shown'));
        try {{ Graph.zoomToFit(700, 40); }} catch (e) {{}}
      }});

      document.getElementById('btnResetView').addEventListener('click', () => {{
        state.search = '';
        state.selectedNodeId = null;
        state.showLabels = false;
        state.labelTitles = false;
        state.focusMode = false;
        state.yearMin = yearMinAll;
        state.yearMax = yearMaxAll;
        state.buckets = new Set(allBuckets);
        state.tiers = new Set(allTiers);
        state.edgeTypes = new Set(allEdgeTypes);
        elSearchBox.value = '';
        elToggleLabels.checked = false;
        elToggleTitles.checked = false;
        elToggleFocus.checked = false;
        elGotoNodeId.value = '';
        elNodePicker.innerHTML = '';
        elYearMin.value = String(yearMinAll);
        elYearMax.value = String(yearMaxAll);
        renderChecklists();
        elDetails.textContent = 'Click a node to see details.';
        elNeighbours.innerHTML = '';
        Graph.zoomToFit(600, 40);
        setEvent('event: reset');
        refresh();
      }});

      // Initial fit
      setTimeout(() => Graph.zoomToFit(600, 40), 200);
    </script>
  </body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-json", type=str, required=True)
    parser.add_argument("--output-html", type=str, required=True)
    parser.add_argument("--title", type=str, default="Interactive Connection Graph (2D)")
    args = parser.parse_args(argv)

    graph_json_path = Path(args.graph_json)
    output_html_path = Path(args.output_html)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _read_json(graph_json_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected graph JSON root: {graph_json_path}")

    graph_data = build_graph_data(raw)
    html = render_html(graph_data, args.title)

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Wrote: {output_html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

