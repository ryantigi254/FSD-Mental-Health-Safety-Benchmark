"""
Generate a self-contained interactive 3D connection graph HTML.

Uses `3d-force-graph` (Three.js-based) via CDN so you can open the HTML directly.

Input: connection_graph.json with shape: { "nodes": [...], "edges": [...] }
Output: HTML file with embedded graph data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def build_graph_data(raw: Dict[str, Any]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = raw.get("nodes", [])
    edges: List[Dict[str, Any]] = raw.get("edges", [])

    graph_nodes: List[Dict[str, Any]] = []
    for n in nodes:
        node_id = str(n.get("id", "")).strip()
        if not node_id:
            continue
        graph_nodes.append(
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

    graph_links: List[Dict[str, Any]] = []
    for e in edges:
        src = str(e.get("source", "")).strip()
        tgt = str(e.get("target", "")).strip()
        if not src or not tgt:
            continue
        graph_links.append(
            {
                "source": src,
                "target": tgt,
                "type": _normalise_edge_type(e.get("type", None) if "type" in e else e.get("connection_type", None)),
                "weight": e.get("weight", 1.0),
            }
        )

    return {"nodes": graph_nodes, "links": graph_links}


def render_html(graph_data: Dict[str, Any], title: str) -> str:
    # Self-contained HTML (CDN JS, embedded data).
    # Avoid fetch() so file:// works without a local server.
    graph_data_json = json.dumps(graph_data, ensure_ascii=False)

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
        inset: 0;
      }}
      #hud {{
        position: absolute;
        top: 12px;
        left: 12px;
        padding: 10px 12px;
        background: rgba(0, 0, 0, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        max-width: 520px;
        backdrop-filter: blur(6px);
      }}
      #hud h1 {{
        font-size: 14px;
        margin: 0 0 6px 0;
        font-weight: 650;
      }}
      #hud .row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        font-size: 12px;
        opacity: 0.95;
      }}
      #hud code {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;
        font-size: 11px;
      }}
      #legend {{
        margin-top: 8px;
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
      #help {{
        margin-top: 8px;
        font-size: 12px;
        opacity: 0.9;
      }}
      .section {{
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
      }}
      .section-title {{
        font-size: 12px;
        font-weight: 700;
        margin: 0 0 8px 0;
        opacity: 0.95;
      }}
      .control {{
        font-size: 12px;
        margin: 6px 0;
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
        padding: 8px 10px;
        border-radius: 8px;
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
        gap: 6px 10px;
      }}
      .pill {{
        display: inline-block;
        padding: 2px 8px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 999px;
        font-size: 11px;
        opacity: 0.9;
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
      .btn {{
        appearance: none;
        border: 1px solid rgba(255, 255, 255, 0.14);
        background: rgba(255, 255, 255, 0.06);
        color: #e6e6e6;
        padding: 6px 10px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 12px;
      }}
      .btn:hover {{
        background: rgba(255, 255, 255, 0.10);
      }}
      .btn:active {{
        transform: translateY(1px);
      }}
    </style>
  </head>
  <body>
    <div id="graph"></div>
    <div id="hud">
      <h1>{title}</h1>
      <div class="row small">
        <div class="pill" id="statNodes">nodes: ?</div>
        <div class="pill" id="statEdges">edges: ?</div>
        <div class="pill" id="statYears">years: ?–?</div>
      </div>

      <div class="section">
        <div class="section-title">Search</div>
        <div class="control">
          <input id="searchBox" type="text" placeholder="paper id or title (press Enter)" />
        </div>
        <div class="control grid2">
          <button class="btn" id="btnClearSearch">Clear</button>
          <button class="btn" id="btnResetView">Reset view</button>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Filters</div>
        <div class="control">
          <label><input type="checkbox" id="toggleLabels" /> Show labels</label>
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
        <div id="details" class="small">Click a node to see details.</div>
        <div id="neighbours"></div>
      </div>

      <div id="help" class="section">
        <div class="small"><strong>Controls:</strong> drag node • scroll zoom • right-drag rotate • click focuses</div>
      </div>

      <div id="legend">
        <div class="swatch" style="background:#38bdf8"></div><div>thematic</div>
        <div class="swatch" style="background:#22c55e"></div><div>citation</div>
        <div class="swatch" style="background:#f59e0b"></div><div>semantic</div>
        <div class="swatch" style="background:#a78bfa"></div><div>multi-type (e.g. citation+thematic)</div>
        <div class="swatch" style="background:#94a3b8"></div><div>(missing)</div>
      </div>
    </div>

    <!-- Provide global THREE for label sprites (three-spritetext expects window.THREE). -->
    <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
    <!-- Three.js based force-graph (3d-force-graph). -->
    <script src="https://unpkg.com/3d-force-graph"></script>
    <script src="https://unpkg.com/three-spritetext@1.9.6/dist/three-spritetext.min.js"></script>
    <script>
      const graphData = {graph_data_json};

      const EDGE_COLOURS = {{
        thematic: '#38bdf8',
        citation: '#22c55e',
        semantic: '#f59e0b',
        '(missing)': '#94a3b8'
      }};

      function edgeColour(edgeType) {{
        const t = String(edgeType || '(missing)').trim().toLowerCase();
        if (EDGE_COLOURS[t]) return EDGE_COLOURS[t];
        // multi-type
        if (t.includes('+')) return '#a78bfa';
        return '#94a3b8';
      }}

      const baseNodes = graphData.nodes.map(n => ({{
        ...n,
        year_num: Number(String(n.year || '').replace(/\\D+/g,'')) || null
      }}));
      // IMPORTANT: 3d-force-graph mutates link.source/link.target into node objects.
      // Keep immutable endpoint ids so filtering always works after graphData() calls.
      const baseLinks = graphData.links.map(l => {{
        const source_id = String(l.source || '').trim();
        const target_id = String(l.target || '').trim();
        return {{
          source_id,
          target_id,
          type: String(l.type || '(missing)').trim().toLowerCase(),
          weight: (l.weight === null || l.weight === undefined) ? 1.0 : l.weight
        }};
      }});

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
      const elYearRangeText = document.getElementById('yearRangeText');
      const elYearMin = document.getElementById('yearMin');
      const elYearMax = document.getElementById('yearMax');
      const elEdgeTypeChecks = document.getElementById('edgeTypeChecks');
      const elBucketChecks = document.getElementById('bucketChecks');
      const elTierChecks = document.getElementById('tierChecks');
      const elSearchBox = document.getElementById('searchBox');
      const elToggleLabels = document.getElementById('toggleLabels');
      const elToggleFocus = document.getElementById('toggleFocus');
      const elDetails = document.getElementById('details');
      const elNeighbours = document.getElementById('neighbours');

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
          const id = `edgeType_${{t.replace(/[^a-z0-9_+]/g,'_')}}`;
          mkCheckRow(elEdgeTypeChecks, id, t, state.edgeTypes.has(t), (isOn) => {{
            if (isOn) state.edgeTypes.add(t); else state.edgeTypes.delete(t);
            refresh();
          }});
        }});

        elBucketChecks.innerHTML = '';
        allBuckets.forEach(b => {{
          const id = `bucket_${{b.replace(/[^a-z0-9_]/g,'_')}}`;
          mkCheckRow(elBucketChecks, id, b, state.buckets.has(b), (isOn) => {{
            if (isOn) state.buckets.add(b); else state.buckets.delete(b);
            refresh();
          }});
        }});

        elTierChecks.innerHTML = '';
        allTiers.forEach(t => {{
          const id = `tier_${{t.replace(/[^a-z0-9_]/g,'_')}}`;
          mkCheckRow(elTierChecks, id, t, state.tiers.has(t), (isOn) => {{
            if (isOn) state.tiers.add(t); else state.tiers.delete(t);
            refresh();
          }});
        }});
      }}

      function formatNodeDetails(n) {{
        if (!n) return 'Click a node to see details.';
        const parts = [];
        parts.push(`ID: ${{n.id}}`);
        if (n.title) parts.push(`Title: ${{n.title}}`);
        if (n.bucket) parts.push(`Bucket: ${{n.bucket}}`);
        if (n.tier) parts.push(`Tier: ${{n.tier}}`);
        if (n.year) parts.push(`Year: ${{n.year}}`);
        if (n.priority_score !== null && n.priority_score !== undefined) parts.push(`Priority: ${{n.priority_score}}`);
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

      function applyLabels(isOn) {{
        if (!isOn) {{
          Graph.nodeThreeObject(() => null).nodeThreeObjectExtend(false);
          return;
        }}
        Graph.nodeThreeObject(node => {{
          const SpriteText = window.SpriteText;
          const sprite = new SpriteText(String(node.id));
          sprite.color = '#e6e6e6';
          sprite.textHeight = 7;
          sprite.backgroundColor = 'rgba(0,0,0,0.55)';
          sprite.padding = 4;
          return sprite;
        }}).nodeThreeObjectExtend(true);
      }}

      function refresh() {{
        const yMin = Math.min(state.yearMin, state.yearMax);
        const yMax = Math.max(state.yearMin, state.yearMax);
        elYearRangeText.textContent = `${{yMin}}–${{yMax}}`;

        const filtered = computeFiltered();
        // Rebuild link objects with string endpoints every time (avoid mutated endpoints).
        const linksForGraph = filtered.links.map(l => ({{
          source: l.source_id,
          target: l.target_id,
          type: l.type,
          weight: l.weight
        }}));
        Graph.graphData({{ nodes: filtered.nodes, links: linksForGraph }});

        elStatNodes.textContent = `nodes: ${{filtered.nodes.length}}`;
        elStatEdges.textContent = `edges: ${{filtered.links.length}}`;
        elStatYears.textContent = `years: ${{yMin}}–${{yMax}}`;

        applyLabels(state.showLabels);
      }}

      const Graph = ForceGraph3D()(document.getElementById('graph'))
        .backgroundColor('#0b0e14')
        .graphData({{
          nodes: baseNodes,
          links: baseLinks.map(l => ({{
            source: l.source_id,
            target: l.target_id,
            type: l.type,
            weight: l.weight
          }}))
        }})
        .nodeId('id')
        .nodeLabel(n => {{
          const parts = [
            `ID: ${{n.id}}`,
            n.title ? `Title: ${{n.title}}` : null,
            n.bucket ? `Bucket: ${{n.bucket}}` : null,
            n.tier ? `Tier: ${{n.tier}}` : null,
            n.year ? `Year: ${{n.year}}` : null,
            (n.priority_score !== null && n.priority_score !== undefined) ? `Priority: ${{n.priority_score}}` : null
          ].filter(Boolean);
          return parts.join('\\n');
        }})
        .nodeColor(n => {{
          // bucket-based colouring (stable)
          const bucket = (n.bucket || '').toLowerCase();
          if (bucket.includes('bucket_a')) return '#ef4444';
          if (bucket.includes('bucket_b')) return '#f97316';
          if (bucket.includes('bucket_c')) return '#22c55e';
          if (bucket.includes('bucket_d')) return '#38bdf8';
          if (bucket.includes('clinical')) return '#a78bfa';
          if (bucket.includes('evaluation')) return '#eab308';
          return '#94a3b8';
        }})
        .nodeVal(n => {{
          // scale node size by priority_score when present
          const p = n.priority_score;
          if (p === null || p === undefined) return 3;
          const f = Number(p);
          if (Number.isFinite(f)) return 3 + (f * 1.6);
          return 3;
        }})
        .linkColor(l => edgeColour(l.type))
        .linkOpacity(0.30)
        .linkWidth(l => {{
          const w = Number(l.weight);
          if (Number.isFinite(w)) return 0.6 + Math.min(2.0, w * 0.15);
          return 0.8;
        }})
        .linkDirectionalParticles(0)
        .onNodeClick(node => {{
          state.selectedNodeId = node ? node.id : null;
          elDetails.textContent = formatNodeDetails(node);

          const current = Graph.graphData();
          const neighbours = [];
          if (node && current.links) {{
            current.links.forEach(l => {{
              const srcId = (typeof l.source === 'object' && l.source) ? l.source.id : l.source;
              const tgtId = (typeof l.target === 'object' && l.target) ? l.target.id : l.target;
              if (String(srcId) === String(node.id)) neighbours.push(tgtId);
              else if (String(tgtId) === String(node.id)) neighbours.push(srcId);
            }});
          }}
          const uniq = Array.from(new Set(neighbours)).sort((a,b) => Number(a)-Number(b));
          if (!uniq.length) {{
            elNeighbours.innerHTML = '';
          }} else {{
            elNeighbours.innerHTML = '<div class="small"><strong>Neighbours</strong> (click to focus):</div>' +
              '<div class="small">' + uniq.map(id => `<a href="#" data-id="${{id}}">${{id}}</a>`).join(', ') + '</div>';
            elNeighbours.querySelectorAll('a[data-id]').forEach(a => {{
              a.addEventListener('click', (ev) => {{
                ev.preventDefault();
                const id = ev.currentTarget.getAttribute('data-id');
                const n = (Graph.graphData().nodes || []).find(nn => String(nn.id) === String(id));
                if (n) {{
                  state.selectedNodeId = n.id;
                  elDetails.textContent = formatNodeDetails(n);
                  focusCameraOnNode(n);
                }}
              }});
            }});
          }}

          focusCameraOnNode(node);
          refresh();
        }});

      // Improve initial layout a bit
      Graph.d3Force('charge').strength(-120);
      Graph.d3Force('link').distance(link => 55);

      function focusCameraOnNode(node) {{
        if (!node) return;
        const distance = 120;
        const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
        Graph.cameraPosition(
          {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
          node,
          900
        );
      }}

      function clampYearSliders() {{
        state.yearMin = Number(elYearMin.value);
        state.yearMax = Number(elYearMax.value);
        refresh();
      }}

      renderChecklists();
      elDetails.textContent = 'Click a node to see details.';
      refresh();

      elYearMin.addEventListener('input', clampYearSliders);
      elYearMax.addEventListener('input', clampYearSliders);

      elToggleLabels.addEventListener('change', () => {{
        state.showLabels = !!elToggleLabels.checked;
        applyLabels(state.showLabels);
      }});

      elToggleFocus.addEventListener('change', () => {{
        state.focusMode = !!elToggleFocus.checked;
        refresh();
      }});

      elSearchBox.addEventListener('keydown', (ev) => {{
        if (ev.key !== 'Enter') return;
        state.search = elSearchBox.value || '';
        refresh();
      }});

      document.getElementById('btnClearSearch').addEventListener('click', () => {{
        elSearchBox.value = '';
        state.search = '';
        refresh();
      }});

      document.getElementById('btnResetView').addEventListener('click', () => {{
        state.search = '';
        state.selectedNodeId = null;
        state.showLabels = false;
        state.focusMode = false;
        state.yearMin = yearMinAll;
        state.yearMax = yearMaxAll;
        state.buckets = new Set(allBuckets);
        state.tiers = new Set(allTiers);
        state.edgeTypes = new Set(allEdgeTypes);
        elSearchBox.value = '';
        elToggleLabels.checked = false;
        elToggleFocus.checked = false;
        elYearMin.value = String(yearMinAll);
        elYearMax.value = String(yearMaxAll);
        renderChecklists();
        elDetails.textContent = 'Click a node to see details.';
        elNeighbours.innerHTML = '';
        refresh();
      }});
    </script>
  </body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-json", type=str, required=True)
    parser.add_argument("--output-html", type=str, required=True)
    parser.add_argument("--title", type=str, default="Interactive Connection Graph (Three.js)")
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

