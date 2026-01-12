"""
Double verification for an existing Literature Review run:
- Validates per-paper artefacts exist and are non-empty
- Validates connection graph nodes/edges correspond to processed papers
- Produces plots (matplotlib/seaborn if available) + Mermaid preview

Designed to run against:
Assignment 2/Literature Review/processed/runs/<run_id>/pipeline/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class PaperLocation:
    paper_id: str
    bucket: str
    tier: str

    def notes_path(self, pipeline_dir: Path) -> Path:
        return pipeline_dir / "notes" / self.bucket / self.tier / f"{self.paper_id}.md"

    def extracted_dir(self, pipeline_dir: Path) -> Path:
        return pipeline_dir / "extracted_data" / self.bucket / self.tier / self.paper_id


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(str(value).strip())
    except Exception:
        return None


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _short_title(title: str, max_len: int = 46) -> str:
    t = _normalise_whitespace(title)
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def _mermaid_escape(label: str) -> str:
    # Mermaid node labels break on quotes / brackets / newlines.
    cleaned = label.replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.replace('"', "'").replace("[", "(").replace("]", ")")
    cleaned = _normalise_whitespace(cleaned)
    return cleaned


def _normalise_edge_type(value: Any) -> str:
    """
    Normalise edge type to a stable string.
    - If JSON stores a list like ["citation", "thematic"], output "citation+thematic"
    - If it stores a string, output a cleaned lower-cased string
    """
    if value is None:
        return "(missing)"
    if isinstance(value, list):
        parts = [str(v).strip().lower() for v in value if str(v).strip()]
        parts = sorted(set(parts))
        return "+".join(parts) if parts else "(missing)"
    s = str(value).strip().lower()
    return s if s else "(missing)"


def _timestamp_slug() -> str:
    # Windows-safe (no ':'), stable and sortable.
    return datetime.now().strftime("%Y-%m-%d__%H%M%S")


def load_processed_papers(processing_log_path: Path) -> List[Dict[str, Any]]:
    data = _read_json(processing_log_path)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected processing_log format (expected list): {processing_log_path}")
    return data


def extract_completed_locations(processing_log: List[Dict[str, Any]]) -> List[PaperLocation]:
    locations: List[PaperLocation] = []
    for entry in processing_log:
        if entry.get("status") != "completed":
            continue
        paper_id = str(entry.get("paper_id", "")).strip()
        if not paper_id:
            continue
        # Prefer bucket/tier from graph if present; in processing_log they are embedded in note path.
        note_path_str = ""
        note_artifact = (entry.get("artifacts") or {}).get("note") or {}
        note_path_str = str(note_artifact.get("path", "")).replace("\\", "/")
        m = re.search(r"/notes/(?P<bucket>[^/]+)/(?P<tier>[^/]+)/(?P<paper_id>[^/]+)\.md$", note_path_str)
        if m:
            locations.append(PaperLocation(paper_id=m.group("paper_id"), bucket=m.group("bucket"), tier=m.group("tier")))
            continue

        # Fallback: try to infer bucket/tier from extracted_data path (if present)
        text_meta = (entry.get("artifacts") or {}).get("text_extraction") or {}
        _ = text_meta  # placeholder to keep structure stable
        # If we cannot infer reliably, skip (this should not happen for your existing runs).
    return locations


def verify_paper_artefacts(pipeline_dir: Path, loc: PaperLocation) -> Dict[str, Any]:
    extracted_dir = loc.extracted_dir(pipeline_dir)
    notes_path = loc.notes_path(pipeline_dir)

    required_files = {
        "metadata.json": extracted_dir / "metadata.json",
        "full_text.txt": extracted_dir / "full_text.txt",
        "citations.json": extracted_dir / "citations.json",
        "sections.json": extracted_dir / "sections.json",
        "note.md": notes_path,
    }

    issues: List[str] = []
    file_checks: Dict[str, Dict[str, Any]] = {}

    for name, path in required_files.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        file_checks[name] = {"path": str(path), "exists": exists, "bytes": size}
        if not exists:
            issues.append(f"missing:{name}")
        elif size == 0:
            issues.append(f"empty:{name}")

    # Validate metadata.json shape (if present)
    meta_ok = False
    if required_files["metadata.json"].exists() and required_files["metadata.json"].stat().st_size > 0:
        try:
            meta = _read_json(required_files["metadata.json"])
            meta_ok = isinstance(meta, dict) and bool(str(meta.get("title", "")).strip())
            if not meta_ok:
                issues.append("invalid:metadata.json_missing_title")
            if "authors" in meta and not isinstance(meta["authors"], list):
                issues.append("invalid:metadata.json_authors_not_list")
            if "year" in meta and _safe_int(meta.get("year")) is None:
                issues.append("invalid:metadata.json_year_not_int")
        except Exception as e:
            issues.append(f"invalid:metadata.json_parse_error:{type(e).__name__}")

    # Validate citations.json is JSON list (if present)
    if required_files["citations.json"].exists() and required_files["citations.json"].stat().st_size > 0:
        try:
            citations = _read_json(required_files["citations.json"])
            if not isinstance(citations, list):
                issues.append("invalid:citations.json_not_list")
        except Exception as e:
            issues.append(f"invalid:citations.json_parse_error:{type(e).__name__}")

    # Sanity check text length (if present)
    if required_files["full_text.txt"].exists():
        try:
            text_len = required_files["full_text.txt"].stat().st_size
            if text_len < 2000:
                issues.append(f"warning:full_text_short:{text_len}_bytes")
        except Exception:
            pass

    return {
        "paper_id": loc.paper_id,
        "bucket": loc.bucket,
        "tier": loc.tier,
        "ok": len([i for i in issues if not i.startswith("warning:")]) == 0,
        "issues": issues,
        "files": file_checks,
        "metadata_ok": meta_ok,
    }


def load_connection_graph(graph_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = _read_json(graph_path)
    if not isinstance(data, dict) or "nodes" not in data or "edges" not in data:
        raise ValueError(f"Unexpected connection_graph format: {graph_path}")
    nodes = data["nodes"]
    edges = data["edges"]
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"Unexpected connection_graph nodes/edges types: {graph_path}")
    return nodes, edges


def verify_graph_consistency(
    pipeline_dir: Path,
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    completed_locs: List[PaperLocation],
) -> Dict[str, Any]:
    completed_ids: Set[str] = {loc.paper_id for loc in completed_locs}
    node_ids: List[str] = [str(n.get("id", "")).strip() for n in graph_nodes]
    node_id_set = {nid for nid in node_ids if nid}

    issues: List[str] = []

    missing_in_graph = sorted(completed_ids - node_id_set, key=lambda x: int(x) if x.isdigit() else x)
    extra_in_graph = sorted(node_id_set - completed_ids, key=lambda x: int(x) if x.isdigit() else x)
    if missing_in_graph:
        issues.append(f"missing_nodes_for_completed:{missing_in_graph}")
    if extra_in_graph:
        issues.append(f"extra_nodes_not_in_completed:{extra_in_graph}")

    edge_endpoint_issues: List[str] = []
    self_loops = 0
    bad_edges = 0
    edge_type_counter = Counter()

    # Build adjacency to test connectivity without relying on external libs.
    adjacency: Dict[str, Set[str]] = defaultdict(set)

    for e in graph_edges:
        src = str(e.get("source", "")).strip()
        tgt = str(e.get("target", "")).strip()
        if not src or not tgt:
            bad_edges += 1
            edge_endpoint_issues.append("edge_missing_source_or_target")
            continue
        if src not in node_id_set or tgt not in node_id_set:
            bad_edges += 1
            edge_endpoint_issues.append(f"edge_endpoint_not_in_nodes:{src}->{tgt}")
            continue
        if src == tgt:
            self_loops += 1

        edge_type = _normalise_edge_type(e.get("type", None) if "type" in e else e.get("connection_type", None))
        edge_type_counter[edge_type] += 1

        adjacency[src].add(tgt)
        adjacency[tgt].add(src)

        w = e.get("weight")
        if w is not None:
            try:
                float(w)
            except Exception:
                edge_endpoint_issues.append(f"edge_weight_not_float:{src}->{tgt}:{w}")

    # Connectivity: count connected components
    seen: Set[str] = set()
    components: List[List[str]] = []
    for nid in node_id_set:
        if nid in seen:
            continue
        stack = [nid]
        comp: List[str] = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            for nxt in adjacency.get(cur, set()):
                if nxt not in seen:
                    stack.append(nxt)
        components.append(sorted(comp, key=lambda x: int(x) if x.isdigit() else x))

    components.sort(key=len, reverse=True)

    # Verify each node maps to an on-disk paper folder and note
    loc_by_id = {loc.paper_id: loc for loc in completed_locs}
    missing_paper_dirs = []
    missing_note_files = []
    for nid in sorted(node_id_set, key=lambda x: int(x) if x.isdigit() else x):
        loc = loc_by_id.get(nid)
        if not loc:
            continue
        if not loc.extracted_dir(pipeline_dir).exists():
            missing_paper_dirs.append(nid)
        if not loc.notes_path(pipeline_dir).exists():
            missing_note_files.append(nid)

    if missing_paper_dirs:
        issues.append(f"graph_nodes_missing_extracted_dirs:{missing_paper_dirs}")
    if missing_note_files:
        issues.append(f"graph_nodes_missing_notes:{missing_note_files}")
    if bad_edges:
        issues.append(f"bad_edges:{bad_edges}")

    return {
        "node_count": len(node_id_set),
        "edge_count": len(graph_edges),
        "bad_edges": bad_edges,
        "self_loops": self_loops,
        "edge_type_counts": dict(edge_type_counter),
        "connected_components": {"count": len(components), "sizes": [len(c) for c in components], "largest": components[0] if components else []},
        "mismatch": {"missing_in_graph": missing_in_graph, "extra_in_graph": extra_in_graph},
        "edge_endpoint_issues_sample": edge_endpoint_issues[:50],
        "issues": issues,
    }


def write_mermaid_preview(
    output_path: Path,
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    max_edges: int,
) -> None:
    # NOTE: Your priority list may contain duplicates for the same paper (e.g. IDs 2 and 4).
    # For Mermaid preview readability, we collapse duplicates by (normalised title, year).
    node_title: Dict[str, str] = {}
    node_year: Dict[str, Optional[int]] = {}
    for n in graph_nodes:
        pid = str(n.get("id", "")).strip()
        title = str(n.get("title", "")).strip()
        node_title[pid] = title
        node_year[pid] = _safe_int(n.get("year"))

    def _canonical_key(pid: str) -> Tuple[str, Optional[int]]:
        t = _normalise_whitespace(node_title.get(pid, "")).lower()
        y = node_year.get(pid)
        return (t, y)

    # Build canonical mapping: pick the lowest numeric id per key as canonical.
    by_key: Dict[Tuple[str, Optional[int]], List[str]] = defaultdict(list)
    for pid in node_title.keys():
        k = _canonical_key(pid)
        if k[0]:  # only if we have a title
            by_key[k].append(pid)

    canonical_of: Dict[str, str] = {}
    aliases_of: Dict[str, List[str]] = defaultdict(list)
    for k, ids in by_key.items():
        # sort with numeric ids first (stable)
        ids_sorted = sorted(ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        canonical = ids_sorted[0]
        for pid in ids_sorted:
            canonical_of[pid] = canonical
        if len(ids_sorted) > 1:
            aliases_of[canonical] = ids_sorted[1:]

    # Rank edges by numeric weight if present, else keep first N.
    ranked_edges: List[Tuple[float, Dict[str, Any]]] = []
    for e in graph_edges:
        w = e.get("weight")
        weight = 1.0
        if w is not None:
            try:
                weight = float(w)
            except Exception:
                weight = 1.0
        ranked_edges.append((weight, e))
    ranked_edges.sort(key=lambda x: x[0], reverse=True)
    picked = [e for _, e in ranked_edges[:max_edges]]

    lines: List[str] = []
    lines.append("```mermaid")
    lines.append("graph LR")

    # Define nodes used
    used_nodes: Set[str] = set()
    for e in picked:
        used_nodes.add(str(e.get("source", "")).strip())
        used_nodes.add(str(e.get("target", "")).strip())

    # Remap used nodes to canonical ids (so 2/4 collapse)
    used_nodes = {canonical_of.get(n, n) for n in used_nodes if n}

    for nid in sorted([n for n in used_nodes if n], key=lambda x: int(x) if x.isdigit() else x):
        t = node_title.get(nid, "")
        aka = aliases_of.get(nid, [])
        aka_str = f" (aka {', '.join(aka)})" if aka else ""
        label = _mermaid_escape(f"{nid}: {_short_title(t)}{aka_str}" if t else nid)
        lines.append(f'  P{nid}["{label}"]')

    # Remap + de-duplicate edges after collapsing aliases (keep max weight per type between nodes)
    collapsed: Dict[Tuple[str, str, str], float] = {}
    for e in picked:
        src_raw = str(e.get("source", "")).strip()
        tgt_raw = str(e.get("target", "")).strip()
        src = canonical_of.get(src_raw, src_raw)
        tgt = canonical_of.get(tgt_raw, tgt_raw)
        if not src or not tgt:
            continue
        if src == tgt:
            continue
        edge_type = _normalise_edge_type(e.get("type", None) if "type" in e else e.get("connection_type", None))
        w = e.get("weight")
        weight = 1.0
        if w is not None:
            try:
                weight = float(w)
            except Exception:
                weight = 1.0

        a, b = (src, tgt) if src <= tgt else (tgt, src)
        key = (a, b, edge_type)
        collapsed[key] = max(collapsed.get(key, 0.0), weight)

    for (src, tgt, edge_type), weight in sorted(
        collapsed.items(), key=lambda kv: (-kv[1], kv[0][2], kv[0][0], kv[0][1])
    ):
        w_str = f" {weight:.2f}" if weight is not None else ""
        label = _mermaid_escape(f"{edge_type}{w_str}")
        lines.append(f"  P{src} -->|{label}| P{tgt}")

    lines.append("```")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_plots(
    output_dir: Path,
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a simple adjacency for degree + a NetworkX graph if available.
    node_ids: List[str] = [str(n.get("id", "")).strip() for n in graph_nodes if str(n.get("id", "")).strip()]
    node_set = set(node_ids)
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    edge_types = []

    for e in graph_edges:
        s = str(e.get("source", "")).strip()
        t = str(e.get("target", "")).strip()
        if not s or not t:
            continue
        if s not in node_set or t not in node_set:
            continue
        adjacency[s].add(t)
        adjacency[t].add(s)
        edge_types.append(_normalise_edge_type(e.get("type", None) if "type" in e else e.get("connection_type", None)))

    degrees = {nid: len(adjacency.get(nid, set())) for nid in node_set}
    degree_values = list(degrees.values())
    edge_type_counts = Counter(edge_types)

    artefacts: Dict[str, str] = {}

    # Degree distribution (seaborn if possible)
    try:
        import matplotlib.pyplot as plt

        try:
            import seaborn as sns

            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(10, 5))
            sns.histplot(degree_values, bins=min(20, max(5, len(set(degree_values)))), kde=False)
            plt.title("Connection Graph Degree Distribution")
            plt.xlabel("Degree")
            plt.ylabel("Count")
            out = output_dir / "degree_distribution.png"
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            artefacts["degree_distribution_png"] = str(out)
        except Exception:
            plt.figure(figsize=(10, 5))
            plt.hist(degree_values, bins=min(20, max(5, len(set(degree_values)))))
            plt.title("Connection Graph Degree Distribution")
            plt.xlabel("Degree")
            plt.ylabel("Count")
            out = output_dir / "degree_distribution.png"
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            artefacts["degree_distribution_png"] = str(out)

        # Edge type counts bar
        plt.figure(figsize=(10, 5))
        labels = list(edge_type_counts.keys())
        values = [edge_type_counts[k] for k in labels]
        plt.bar(labels, values)
        plt.title("Edge Type Counts")
        plt.xlabel("Edge type")
        plt.ylabel("Count")
        plt.xticks(rotation=25, ha="right")
        out = output_dir / "edge_type_counts.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        artefacts["edge_type_counts_png"] = str(out)

        # Layout plot (NetworkX if available)
        try:
            import networkx as nx

            g = nx.Graph()
            for nid in node_set:
                g.add_node(nid)
            for e in graph_edges:
                s = str(e.get("source", "")).strip()
                t = str(e.get("target", "")).strip()
                if not s or not t:
                    continue
                if s not in node_set or t not in node_set:
                    continue
                g.add_edge(s, t)

            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(g, seed=42, k=1.25 / math.sqrt(max(1, g.number_of_nodes())))
            nx.draw_networkx_nodes(g, pos, node_size=250, alpha=0.85)
            nx.draw_networkx_edges(g, pos, alpha=0.25, width=0.8)
            nx.draw_networkx_labels(g, pos, font_size=7)
            plt.title("Connection Graph Layout (Spring)")
            plt.axis("off")
            out = output_dir / "connection_graph_layout.png"
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            artefacts["connection_graph_layout_png"] = str(out)
        except Exception:
            # If networkx isn't installed in the current env, skip this plot.
            pass

    except Exception:
        # matplotlib isn't available; no PNG plots.
        pass

    return artefacts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-dir",
        type=str,
        required=True,
        help="Path to run pipeline dir (contains processing_log.json, connection_graph.json, notes/, extracted_data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write verification outputs (default: <pipeline>/verification/<timestamp>/)",
    )
    parser.add_argument("--max-mermaid-edges", type=int, default=60)
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args(argv)

    pipeline_dir = Path(args.pipeline_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (pipeline_dir / "verification" / _timestamp_slug())
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    processing_log_path = pipeline_dir / "processing_log.json"
    connection_graph_path = pipeline_dir / "connection_graph.json"

    if not processing_log_path.exists():
        raise FileNotFoundError(f"processing_log.json not found: {processing_log_path}")
    if not connection_graph_path.exists():
        raise FileNotFoundError(f"connection_graph.json not found: {connection_graph_path}")

    processing_log = load_processed_papers(processing_log_path)
    completed_locs = extract_completed_locations(processing_log)

    paper_results = [verify_paper_artefacts(pipeline_dir, loc) for loc in completed_locs]
    paper_ok = sum(1 for r in paper_results if r["ok"])
    paper_issues = [r for r in paper_results if not r["ok"]]

    nodes, edges = load_connection_graph(connection_graph_path)
    graph_check = verify_graph_consistency(pipeline_dir, nodes, edges, completed_locs)

    mermaid_path = output_dir / "connection_graph_preview.md"
    write_mermaid_preview(mermaid_path, nodes, edges, max_edges=args.max_mermaid_edges)

    plot_outputs = write_plots(output_dir, nodes, edges)

    report = {
        "pipeline_dir": str(pipeline_dir),
        "generated_outputs_in": str(output_dir),
        "papers": {
            "completed_count": len(completed_locs),
            "ok_count": paper_ok,
            "not_ok_count": len(paper_issues),
            "not_ok": paper_issues,
        },
        "graph": graph_check,
        "generated": {
            "mermaid_preview": str(mermaid_path),
            "plots": plot_outputs,
        },
    }

    report_path = output_dir / "double_verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    severe_missing = any(
        any(issue.startswith("missing:") or issue.startswith("empty:") or issue.startswith("invalid:") for issue in r["issues"])
        for r in paper_results
    )
    graph_has_issues = bool(graph_check.get("issues"))
    fail = args.fail_on_missing and (severe_missing or graph_has_issues)

    print(f"[OK] Wrote report: {report_path}")
    print(f"[OK] Mermaid preview: {mermaid_path}")
    for k, v in plot_outputs.items():
        print(f"[OK] Plot: {k} -> {v}")
    print(f"[INFO] Papers completed={len(completed_locs)} ok={paper_ok} not_ok={len(paper_issues)}")
    print(f"[INFO] Graph nodes={graph_check['node_count']} edges={graph_check['edge_count']} components={graph_check['connected_components']['count']}")

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

