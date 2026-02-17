"""
Generate a self-contained interactive 2D connection graph HTML using SVG + d3-force.

Why SVG (vs canvas):
- Real DOM nodes => reliable click/hover/drag in normal browsers.
- Debuggable styling and accessible interactions.

Input:
  --graph-json path/to/connection_graph.json
Output:
  --output-html path/to/output.html
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
                "year_num": _safe_int(n.get("year")),
                "priority_score": n.get("priority_score", None),
            }
        )

    # Collapse duplicates by (normalised title, year int)
    by_key: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {}
    for n in raw_nodes:
        key = (_normalise_title(n.get("title")), _safe_int(n.get("year")))
        if not key[0]:
            key = (f"__id__{n['id']}", key[1])
        by_key.setdefault(key, []).append(n)

    canonical_of: Dict[str, str] = {}
    graph_nodes: List[Dict[str, Any]] = []
    for _key, group in by_key.items():
        ids = _sort_paper_ids([g["id"] for g in group])
        canonical_id = ids[0]
        for g in group:
            canonical_of[g["id"]] = canonical_id

        # Prefer canonical id, but keep best payload if it has richer data
        canonical_node = None
        best_score = -1.0
        for g in group:
            score = g.get("priority_score")
            try:
                score_f = float(score) if score is not None else -1.0
            except Exception:
                score_f = -1.0
            if g["id"] == canonical_id:
                score_f += 0.001
            if score_f > best_score:
                best_score = score_f
                canonical_node = g

        assert canonical_node is not None
        out = dict(canonical_node)
        out["aliases"] = [i for i in ids[1:]]
        graph_nodes.append(out)

    # Remap + de-duplicate links after collapsing: key by undirected (a,b,type)
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
    for (src, tgt, edge_type), w_f in sorted(
        collapsed_links.items(), key=lambda kv: (-kv[1], kv[0][2], kv[0][0], kv[0][1])
    ):
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

    # Avoid JS template literals here to prevent f-string brace conflicts.
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

      #nodePicker a, #neighbours a {{
        color: #93c5fd;
        text-decoration: none;
      }}
      #nodePicker a:hover, #neighbours a:hover {{
        text-decoration: underline;
      }}

      /* SVG graph styling */
      svg {{
        width: 100%;
        height: 100%;
        display: block;
        touch-action: none;
        user-select: none;
      }}

      .link {{
        stroke: rgba(147, 197, 253, 0.22);
        stroke-width: 1;
      }}

      .link.citation {{ stroke: rgba(34, 197, 94, 0.35); }}
      .link.semantic {{ stroke: rgba(245, 158, 11, 0.35); }}
      .link.thematic {{ stroke: rgba(56, 189, 248, 0.35); }}
      .link.multi {{ stroke: rgba(167, 139, 250, 0.30); }}

      .node circle {{
        stroke: rgba(0,0,0,0.55);
        stroke-width: 1.2px;
        cursor: grab;
      }}

      .node.dragging circle {{
        cursor: grabbing;
      }}

      .node.selected circle {{
        stroke: rgba(255,255,255,0.95);
        stroke-width: 2.6px;
      }}

      .node text {{
        font-size: 11px;
        fill: rgba(230,230,230,0.95);
        paint-order: stroke;
        stroke: rgba(0,0,0,0.75);
        stroke-width: 3px;
        stroke-linejoin: round;
        pointer-events: none;
      }}

      body.panel-collapsed #panel {{
        display: none;
      }}
      body.panel-collapsed #graph {{
        inset: 0;
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
        <div id="nodePicker" class="small" style="margin-top:6px;"></div>
        <div id="details" class="small">Click a node to see details.</div>
        <div id="neighbours"></div>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
    <script>
      const BUILD_ID = "{build_id}";
      const graphData = {graph_data_json};

      const EDGE_COLOURS = {{
        thematic: 'rgba(56,189,248,0.35)',
        citation: 'rgba(34,197,94,0.35)',
        semantic: 'rgba(245,158,11,0.35)',
        '(missing)': 'rgba(148,163,184,0.25)'
      }};

      function edgeClass(edgeType) {{
        const t = String(edgeType || '(missing)').trim().toLowerCase();
        if (t.includes('+')) return 'multi';
        if (t === 'citation' || t === 'semantic' || t === 'thematic') return t;
        return '';
      }}

      function nodeFill(bucket) {{
        const b = String(bucket || '').toLowerCase();
        if (b.includes('bucket_a')) return '#ef4444';
        if (b.includes('bucket_b')) return '#f97316';
        if (b.includes('bucket_c')) return '#22c55e';
        if (b.includes('bucket_d')) return '#38bdf8';
        if (b.includes('clinical')) return '#a78bfa';
        if (b.includes('evaluation')) return '#eab308';
        if (b.includes('method')) return '#94a3b8';
        if (b.includes('survey')) return '#64748b';
        if (b.includes('resource')) return '#fb7185';
        return '#94a3b8';
      }}

      function shortTitle(text, maxLen) {{
        const s = String(text || '').trim();
        if (!s) return '';
        const n = maxLen || 58;
        return s.length > n ? (s.slice(0, n - 1) + '…') : s;
      }}

      const baseNodes = (graphData.nodes || []).map(n => Object.assign({{}}, n, {{
        year_num: Number.isFinite(Number(n.year_num)) ? Number(n.year_num) : Number(n.year) || null
      }}));

      const baseLinks = (graphData.links || []).map(l => ({{
        source_id: String(l.source_id || l.source || '').trim(),
        target_id: String(l.target_id || l.target || '').trim(),
        type: String(l.type || '(missing)').trim().toLowerCase(),
        weight: (l.weight === null || l.weight === undefined) ? 1.0 : Number(l.weight)
      }})).filter(l => l.source_id && l.target_id && l.source_id !== l.target_id);

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

      const elStatNodes = document.getElementById('statNodes');
      const elStatEdges = document.getElementById('statEdges');
      const elStatYears = document.getElementById('statYears');
      const elStatBuild = document.getElementById('statBuild');
      const elStatEvent = document.getElementById('statEvent');
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
      const elGotoNodeId = document.getElementById('gotoNodeId');
      const elBtnGotoNode = document.getElementById('btnGotoNode');
      const elNodePicker = document.getElementById('nodePicker');
      const elDetails = document.getElementById('details');
      const elNeighbours = document.getElementById('neighbours');

      function setEvent(message) {{
        if (!elStatEvent) return;
        elStatEvent.textContent = String(message || '');
        elStatEvent.setAttribute('aria-label', elStatEvent.textContent);
      }}

      function clampRangePair(minEl, maxEl) {{
        const a = Number(minEl.value);
        const b = Number(maxEl.value);
        if (a <= b) return [a, b];
        return [b, a];
      }}

      elYearMin.min = String(yearMinAll);
      elYearMin.max = String(yearMaxAll);
      elYearMax.min = String(yearMinAll);
      elYearMax.max = String(yearMaxAll);
      elYearMin.value = String(yearMinAll);
      elYearMax.value = String(yearMaxAll);

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
          const id = 'edgeType_' + t.replace(/[^a-z0-9_+]/g,'_');
          mkCheckRow(elEdgeTypeChecks, id, t, state.edgeTypes.has(t), (isOn) => {{
            if (isOn) state.edgeTypes.add(t); else state.edgeTypes.delete(t);
            setEvent('event: edge type ' + t + ' -> ' + (isOn ? 'on' : 'off'));
            refresh();
          }});
        }});

        elBucketChecks.innerHTML = '';
        allBuckets.forEach(b => {{
          const id = 'bucket_' + b.replace(/[^a-z0-9_]/g,'_');
          mkCheckRow(elBucketChecks, id, b, state.buckets.has(b), (isOn) => {{
            if (isOn) state.buckets.add(b); else state.buckets.delete(b);
            setEvent('event: bucket ' + b + ' -> ' + (isOn ? 'on' : 'off'));
            refresh();
          }});
        }});

        elTierChecks.innerHTML = '';
        allTiers.forEach(t => {{
          const id = 'tier_' + t.replace(/[^a-z0-9_]/g,'_');
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
        parts.push('ID: ' + n.id);
        if (n.aliases && n.aliases.length) parts.push('Also listed as: ' + n.aliases.join(', '));
        if (n.title) parts.push('Title: ' + n.title);
        if (n.bucket) parts.push('Bucket: ' + n.bucket);
        if (n.tier) parts.push('Tier: ' + n.tier);
        if (n.year) parts.push('Year: ' + n.year);
        if (n.priority_score !== null && n.priority_score !== undefined) parts.push('Priority: ' + n.priority_score);
        return parts.join('\\n');
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
            const hay = (String(n.id) + ' ' + String(n.title || '')).toLowerCase();
            if (!hay.includes(search)) return false;
          }}
          return true;
        }});

        const nodeIdSet = new Set(nodes.map(n => String(n.id)));
        let links = baseLinks.filter(l => {{
          if (!nodeIdSet.has(l.source_id) || !nodeIdSet.has(l.target_id)) return false;
          if (!state.edgeTypes.has(l.type)) return false;
          return true;
        }});

        if (state.focusMode && state.selectedNodeId && nodeIdSet.has(String(state.selectedNodeId))) {{
          const keep = new Set([String(state.selectedNodeId)]);
          links.forEach(l => {{
            if (l.source_id === String(state.selectedNodeId)) keep.add(l.target_id);
            if (l.target_id === String(state.selectedNodeId)) keep.add(l.source_id);
          }});
          nodes = nodes.filter(n => keep.has(String(n.id)));
          const keepIds = new Set(nodes.map(n => String(n.id)));
          links = links.filter(l => keepIds.has(l.source_id) && keepIds.has(l.target_id));
        }}

        return {{ nodes, links }};
      }}

      function buildAdjacency(links) {{
        const adj = new Map();
        (links || []).forEach(l => {{
          const s = String(l.source_id);
          const t = String(l.target_id);
          if (!adj.has(s)) adj.set(s, new Set());
          if (!adj.has(t)) adj.set(t, new Set());
          adj.get(s).add(t);
          adj.get(t).add(s);
        }});
        return adj;
      }}

      function computeDegree(links) {{
        const d = new Map();
        (links || []).forEach(l => {{
          const s = String(l.source_id);
          const t = String(l.target_id);
          d.set(s, (d.get(s) || 0) + 1);
          d.set(t, (d.get(t) || 0) + 1);
        }});
        return d;
      }}

      function renderNodePicker(nodes, links) {{
        const degree = computeDegree(links);
        const sorted = (nodes || []).slice().sort((a, b) => {{
          const da = degree.get(String(a.id)) || 0;
          const db = degree.get(String(b.id)) || 0;
          if (db !== da) return db - da;
          return String(a.id).localeCompare(String(b.id), undefined, {{ numeric: true }});
        }});

        const maxItems = 18;
        const subset = sorted.slice(0, maxItems);
        const rows = subset.map(n => {{
          const id = String(n.id);
          const label = id + (n.title ? (': ' + shortTitle(n.title, 58)) : '');
          return '<a href="#" data-node-id="' + id + '">' + label + '</a>';
        }});
        if (sorted.length > maxItems) {{
          rows.push('<div style="opacity:0.75; margin-top:6px;">…and ' + String(sorted.length - maxItems) + ' more (use search)</div>');
        }}
        elNodePicker.innerHTML = rows.join('<br/>');
        elNodePicker.querySelectorAll('a[data-node-id]').forEach(a => {{
          a.addEventListener('click', (ev) => {{
            ev.preventDefault();
            const id = ev.currentTarget.getAttribute('data-node-id');
            if (!id) return;
            selectNodeById(id);
          }});
        }});
      }}

      function renderNeighboursForSelected(selectedId, nodes, links) {{
        const id = String(selectedId || '').trim();
        if (!id) {{
          elNeighbours.innerHTML = '';
          return;
        }}
        const adj = buildAdjacency(links);
        const neigh = Array.from(adj.get(id) || []);
        neigh.sort((a, b) => Number(a) - Number(b));
        if (!neigh.length) {{
          elNeighbours.innerHTML = '';
          return;
        }}
        const nodeById = new Map((nodes || []).map(n => [String(n.id), n]));
        const fmt = (nid) => {{
          const n = nodeById.get(String(nid));
          if (!n) return String(nid);
          const t = shortTitle(n.title, 48);
          return t ? (String(n.id) + ': ' + t) : String(n.id);
        }};
        elNeighbours.innerHTML =
          '<div class="small"><strong>Neighbours</strong> (click to focus):</div>' +
          '<div class="small">' +
          neigh.map(nid => '<a href="#" data-id="' + String(nid) + '">' + fmt(nid) + '</a>').join('<br/>') +
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

      // SVG setup
      const graphEl = document.getElementById('graph');
      const svg = d3.select(graphEl).append('svg');
      const rootG = svg.append('g');
      const linkG = rootG.append('g').attr('class', 'links');
      const nodeG = rootG.append('g').attr('class', 'nodes');

      const zoom = d3.zoom()
        .scaleExtent([0.2, 6])
        .on('zoom', (ev) => {{
          rootG.attr('transform', ev.transform);
          const k = ev.transform.k;
          // show labels only when zoomed in enough or toggle is on
          const show = state.showLabels || k >= 1.6;
          nodeG.selectAll('text').style('display', show ? null : 'none');
        }});
      svg.call(zoom);

      let simulation = null;
      let currentNodes = [];
      let currentLinks = [];
      let currentAdj = new Map();

      function nodeRadius(n) {{
        const p = Number(n.priority_score);
        const score = Number.isFinite(p) ? Math.min(6, Math.max(0, p)) : 0;
        return 4.0 + score * 0.55;
      }}

      function linkWidth(l) {{
        const w = Number(l.weight);
        if (Number.isFinite(w)) return 0.6 + Math.min(2.8, w * 0.18);
        return 0.8;
      }}

      function updateGraph(nodes, links) {{
        currentNodes = (nodes || []).map(n => Object.assign({{}}, n));
        currentLinks = (links || []).map(l => Object.assign({{}}, l));
        currentAdj = buildAdjacency(currentLinks);

        const nodeById = new Map(currentNodes.map(n => [String(n.id), n]));
        const simLinks = currentLinks.map(l => ({{
          source: nodeById.get(String(l.source_id)),
          target: nodeById.get(String(l.target_id)),
          type: l.type,
          weight: l.weight,
          source_id: l.source_id,
          target_id: l.target_id
        }})).filter(l => l.source && l.target);

        // Links
        const linkSel = linkG.selectAll('line').data(simLinks, d => String(d.source_id) + '->' + String(d.target_id) + '|' + String(d.type));
        linkSel.exit().remove();
        const linkEnter = linkSel.enter().append('line')
          .attr('class', d => 'link ' + edgeClass(d.type))
          .attr('stroke-width', d => linkWidth(d));
        const linkAll = linkEnter.merge(linkSel);

        // Nodes
        const nodeSel = nodeG.selectAll('g.node').data(currentNodes, d => String(d.id));
        nodeSel.exit().remove();

        const nodeEnter = nodeSel.enter().append('g').attr('class', 'node')
          .call(d3.drag()
            .on('start', (ev, d) => {{
              setEvent('event: drag ' + d.id);
              d3.select(ev.sourceEvent && ev.sourceEvent.target ? ev.sourceEvent.target.closest('g.node') : ev.subject).classed('dragging', true);
              if (!ev.active && simulation) simulation.alphaTarget(0.25).restart();
              d.fx = d.x;
              d.fy = d.y;
            }})
            .on('drag', (ev, d) => {{
              d.fx = ev.x;
              d.fy = ev.y;
            }})
            .on('end', (ev, d) => {{
              setEvent('event: drag end ' + d.id);
              d3.select(ev.sourceEvent && ev.sourceEvent.target ? ev.sourceEvent.target.closest('g.node') : ev.subject).classed('dragging', false);
              if (!ev.active && simulation) simulation.alphaTarget(0);
              d.fx = null;
              d.fy = null;
            }})
          );

        nodeEnter.append('circle')
          .attr('r', d => nodeRadius(d))
          .attr('fill', d => nodeFill(d.bucket));

        nodeEnter.append('text')
          .attr('dy', -8)
          .attr('text-anchor', 'middle')
          .text(d => {{
            const label = state.labelTitles && d.title ? shortTitle(d.title, 42) : String(d.id);
            return label;
          }});

        nodeEnter.on('click', (ev, d) => {{
          ev.stopPropagation();
          selectNodeById(String(d.id));
        }});

        const nodeAll = nodeEnter.merge(nodeSel);

        // Hover highlighting
        nodeAll.on('mouseenter', (ev, d) => {{
          const hid = String(d.id);
          const neigh = currentAdj.get(hid) || new Set();
          nodeG.selectAll('g.node').style('opacity', nd => {{
            const nid = String(nd.id);
            if (nid === hid) return 1.0;
            return neigh.has(nid) ? 1.0 : 0.18;
          }});
          linkG.selectAll('line').style('opacity', ld => {{
            const s = String(ld.source_id);
            const t = String(ld.target_id);
            return (s === hid || t === hid) ? 0.75 : 0.08;
          }});
        }});
        nodeAll.on('mouseleave', () => {{
          nodeG.selectAll('g.node').style('opacity', null);
          linkG.selectAll('line').style('opacity', null);
        }});

        // Simulation
        if (simulation) simulation.stop();
        const w = graphEl.clientWidth || 800;
        const h = graphEl.clientHeight || 600;

        simulation = d3.forceSimulation(currentNodes)
          .force('link', d3.forceLink(simLinks).id(d => String(d.id)).distance(75).strength(0.45))
          .force('charge', d3.forceManyBody().strength(-220))
          .force('collide', d3.forceCollide().radius(d => nodeRadius(d) + 6))
          .force('center', d3.forceCenter(w / 2, h / 2));

        simulation.on('tick', () => {{
          linkAll
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

          nodeAll.attr('transform', d => 'translate(' + d.x + ',' + d.y + ')');
        }});

        // Hide labels until zoomed in enough (unless toggle)
        const show = state.showLabels;
        nodeG.selectAll('text').style('display', show ? null : 'none');
      }}

      function fitGraph() {{
        const w = graphEl.clientWidth || 800;
        const h = graphEl.clientHeight || 600;
        if (!currentNodes.length) return;
        const xs = currentNodes.map(n => n.x).filter(Number.isFinite);
        const ys = currentNodes.map(n => n.y).filter(Number.isFinite);
        if (!xs.length || !ys.length) return;
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minY = Math.min(...ys), maxY = Math.max(...ys);
        const pad = 40;
        const dx = (maxX - minX) || 1;
        const dy = (maxY - minY) || 1;
        const scale = Math.max(0.25, Math.min(4.5, 0.9 / Math.max(dx / (w - pad), dy / (h - pad))));
        const tx = (w / 2) - scale * (minX + maxX) / 2;
        const ty = (h / 2) - scale * (minY + maxY) / 2;
        svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
      }}

      function selectNodeById(nodeId) {{
        const id = String(nodeId || '').trim();
        if (!id) return;
        const n = currentNodes.find(nn => String(nn.id) === id);
        if (!n) {{
          setEvent('event: select node ' + id + ' (not in current filter)');
          return;
        }}
        state.selectedNodeId = n.id;
        elDetails.textContent = formatNodeDetails(n);
        setEvent('event: select node ' + id);
        renderNeighboursForSelected(id, currentNodes, currentLinks);

        nodeG.selectAll('g.node').classed('selected', d => String(d.id) === id);
        if (simulation) simulation.alphaTarget(0.12).restart();
        setTimeout(() => {{ if (simulation) simulation.alphaTarget(0); }}, 450);

        // centre view on node
        const w = graphEl.clientWidth || 800;
        const h = graphEl.clientHeight || 600;
        const t = d3.zoomTransform(svg.node());
        const k = t.k || 1;
        const tx = w / 2 - k * n.x;
        const ty = h / 2 - k * n.y;
        svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(Math.max(1.6, k)));
      }}

      function refresh() {{
        if (elStatBuild) elStatBuild.textContent = 'build: ' + BUILD_ID;
        if (elStatBuild) elStatBuild.setAttribute('aria-label', elStatBuild.textContent);

        // slider ordering
        if (Number(elYearMin.value) > Number(elYearMax.value)) {{
          elYearMax.value = String(elYearMin.value);
        }}
        state.yearMin = Number(elYearMin.value);
        state.yearMax = Number(elYearMax.value);
        const yMin = Math.min(state.yearMin, state.yearMax);
        const yMax = Math.max(state.yearMin, state.yearMax);
        elYearRangeText.textContent = String(yMin) + '–' + String(yMax);

        const filtered = computeFiltered();
        elStatNodes.textContent = 'nodes: ' + String(filtered.nodes.length);
        elStatEdges.textContent = 'edges: ' + String(filtered.links.length);
        elStatYears.textContent = 'years: ' + String(yMin) + '–' + String(yMax);
        elStatNodes.setAttribute('aria-label', elStatNodes.textContent);
        elStatEdges.setAttribute('aria-label', elStatEdges.textContent);
        elStatYears.setAttribute('aria-label', elStatYears.textContent);

        updateGraph(filtered.nodes, filtered.links);
        renderNodePicker(filtered.nodes, filtered.links);

        // keep selected node if still visible
        if (state.selectedNodeId && !filtered.nodes.some(n => String(n.id) === String(state.selectedNodeId))) {{
          state.selectedNodeId = null;
          elDetails.textContent = 'Click a node to see details.';
          elNeighbours.innerHTML = '';
        }}
      }}

      // UI events
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

      elSearchBox.addEventListener('keydown', (ev) => {{
        if (ev.key !== 'Enter') return;
        state.search = elSearchBox.value || '';
        setEvent('event: search "' + state.search + '"');
        refresh();
      }});

      document.getElementById('btnClearSearch').addEventListener('click', () => {{
        elSearchBox.value = '';
        state.search = '';
        setEvent('event: clear search');
        refresh();
      }});

      document.getElementById('btnFitGraph').addEventListener('click', () => {{
        setEvent('event: fit graph');
        fitGraph();
      }});

      document.getElementById('btnTogglePanel').addEventListener('click', () => {{
        document.body.classList.toggle('panel-collapsed');
        setEvent('event: panel ' + (document.body.classList.contains('panel-collapsed') ? 'collapsed' : 'shown'));
        fitGraph();
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
        elYearMin.value = String(yearMinAll);
        elYearMax.value = String(yearMaxAll);
        renderChecklists();
        elDetails.textContent = 'Click a node to see details.';
        elNeighbours.innerHTML = '';
        setEvent('event: reset');
        refresh();
        fitGraph();
      }});

      elGotoNodeId.addEventListener('keydown', (ev) => {{
        if (ev.key !== 'Enter') return;
        selectNodeById(elGotoNodeId.value || '');
      }});
      elBtnGotoNode.addEventListener('click', () => {{
        selectNodeById(elGotoNodeId.value || '');
      }});

      // click on empty space clears selection
      svg.on('click', () => {{
        state.selectedNodeId = null;
        nodeG.selectAll('g.node').classed('selected', false);
        elDetails.textContent = 'Click a node to see details.';
        elNeighbours.innerHTML = '';
        setEvent('event: deselect');
      }});

      renderChecklists();
      refresh();
      setTimeout(() => fitGraph(), 250);
    </script>
  </body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-json", type=str, required=True)
    parser.add_argument("--output-html", type=str, required=True)
    parser.add_argument("--title", type=str, default="Interactive Connection Graph (2D, SVG)")
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

