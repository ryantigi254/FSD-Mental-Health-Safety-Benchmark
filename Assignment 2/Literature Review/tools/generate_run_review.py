import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_mermaid_label(s: str, max_len: int = 60) -> str:
    t = (s or "").replace("\n", " ").replace("\r", " ").strip()
    t = t.replace('"', "'")
    if len(t) > max_len:
        t = t[: max_len - 3] + "..."
    return t


def _connection_type_str(edge: Dict[str, Any]) -> str:
    v = edge.get("connection_type")
    if isinstance(v, list):
        return "+".join(str(x) for x in v if x) or "unknown"
    if isinstance(v, str):
        return v
    return "unknown"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def _find_processed_ids(pipeline_dir: Path) -> List[str]:
    meta_paths = list(pipeline_dir.glob("extracted_data/**/metadata.json"))
    ids = {p.parent.name for p in meta_paths}

    def _key(x: str):
        return (0, int(x)) if x.isdigit() else (1, x)

    return sorted(ids, key=_key)


def _edge_degree(edges: List[Dict[str, Any]]) -> Counter:
    deg = Counter()
    for e in edges:
        if not isinstance(e, dict):
            continue
        s = str(e.get("source", "")).strip()
        t = str(e.get("target", "")).strip()
        if s and t:
            deg[s] += 1
            deg[t] += 1
    return deg


def _top_semantic_edges(edges: List[Dict[str, Any]], top_n: int = 15) -> List[Tuple[float, str, str]]:
    out: List[Tuple[float, str, str]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        ctype = e.get("connection_type")
        types = set(ctype) if isinstance(ctype, list) else {ctype} if isinstance(ctype, str) else set()
        if "semantic" not in {str(x) for x in types}:
            continue
        sim = e.get("semantic_similarity")
        if sim is None:
            continue
        try:
            out.append((float(sim), str(e.get("source", "")), str(e.get("target", ""))))
        except Exception:
            continue
    out.sort(reverse=True)
    return out[:top_n]


def build_review(run_root: Path, sources_dir: Path, priority_csv: Path) -> Path:
    pipeline = run_root / "pipeline"
    conn_path = pipeline / "connection_graph.json"
    syn_path = pipeline / "synthesis_matrix.csv"
    log_path = pipeline / "processing_log.json"

    processed_ids = _find_processed_ids(pipeline)
    notes_count = len(list(pipeline.glob("notes/**/*.md")))

    # Graph
    graph: Dict[str, Any] = {}
    if conn_path.exists():
        try:
            graph = _load_json(conn_path)
        except Exception:
            graph = {}

    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    node_by_id: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict):
            nid = str(n.get("id", "")).strip()
            if nid:
                node_by_id[nid] = n

    edge_type_counts = Counter(_connection_type_str(e) for e in edges if isinstance(e, dict))
    deg = _edge_degree(edges)
    top_degree = deg.most_common(10)
    sem_edges = _top_semantic_edges(edges, top_n=15)

    # Synthesis
    synthesis_rows = _read_csv_rows(syn_path) if syn_path.exists() else []
    pending_placeholders = sum(
        1
        for r in synthesis_rows
        if (r.get("How it Connects", "") or "").strip() == "Connection analysis pending."
    )

    # Priority list coverage
    priority_rows = _read_csv_rows(priority_csv)
    priority_ids = [str(r.get("Paper_ID", "")).strip() for r in priority_rows if str(r.get("Paper_ID", "")).strip()]
    priority_set = set(priority_ids)
    processed_set = set(processed_ids)

    missing_from_priority = sorted(processed_set - priority_set, key=lambda x: int(x) if x.isdigit() else x)
    not_processed = [pid for pid in priority_ids if pid not in processed_set]

    # PDF coverage
    local_files: List[str] = []
    for r in priority_rows:
        notes = (r.get("Notes") or "")
        if "local_file=" in notes:
            lf = notes.split("local_file=", 1)[1].strip().strip('"').strip("'")
            if lf:
                local_files.append(lf)

    existing_pdfs: List[str] = []
    missing_pdfs: List[str] = []
    for lf in local_files:
        p = sources_dir / Path(lf)
        if p.exists():
            existing_pdfs.append(lf)
        else:
            missing_pdfs.append(lf)

    total_pdfs = len(list(sources_dir.glob("**/*.pdf")))

    # Mermaid snippet (top semantic edges)
    mermaid: List[str] = ["```mermaid", "graph LR"]
    for sim, s, t in sem_edges:
        s_id = str(s).strip()
        t_id = str(t).strip()
        s_title = _safe_mermaid_label((node_by_id.get(s_id, {}) or {}).get("title", ""), max_len=45)
        t_title = _safe_mermaid_label((node_by_id.get(t_id, {}) or {}).get("title", ""), max_len=45)

        s_node = f"P{s_id}['{s_id}: {s_title}' ]" if s_title else f"P{s_id}['{s_id}']"
        t_node = f"P{t_id}['{t_id}: {t_title}' ]" if t_title else f"P{t_id}['{t_id}']"

        mermaid.append(f"  {s_node} -->|sem {sim:.2f}| {t_node}")
    mermaid.append("```")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md: List[str] = []
    md.append(f"# Run Review: {run_root.name}")
    md.append("")
    md.append(f"**Generated:** {now}")
    md.append("")

    md.append("## Summary")
    md.append(f"- **Processed papers (metadata.json found):** {len(processed_ids)}")
    md.append(f"- **Notes generated (.md):** {notes_count}")
    md.append(f"- **Synthesis matrix rows:** {len(synthesis_rows)}")
    md.append(f"- **Synthesis placeholder rows (\"Connection analysis pending.\"):** {pending_placeholders}")
    md.append(f"- **Graph nodes:** {len(nodes)}")
    md.append(f"- **Graph edges:** {len(edges)}")
    md.append("")

    md.append("## Coverage vs Priority List")
    md.append(f"- **Priority list papers:** {len(priority_ids)}")
    md.append(f"- **Processed in this run:** {len(processed_ids)}")
    md.append(f"- **Not processed:** {len(not_processed)}")
    md.append("")

    md.append("### Processed Paper IDs")
    md.append(", ".join(processed_ids) if processed_ids else "(none)")
    md.append("")

    md.append("### Not processed Paper IDs (from priority list)")
    md.append(", ".join(not_processed[:80]) + (" ..." if len(not_processed) > 80 else ""))
    md.append("")

    if missing_from_priority:
        md.append("### Processed IDs not found in PRIORITY_READING_LIST.csv")
        md.append(", ".join(missing_from_priority))
        md.append("")

    md.append("## PDF Coverage (from PRIORITY_READING_LIST.csv Notes local_file=...)")
    md.append(f"- **Entries with local_file=:** {len(local_files)}")
    md.append(f"- **PDFs found under sources/:** {len(existing_pdfs)}")
    md.append(f"- **Missing PDFs referenced by local_file=:** {len(missing_pdfs)}")
    md.append(f"- **Total PDFs present under sources/**:** {total_pdfs}")
    if missing_pdfs:
        md.append("")
        md.append("### Missing referenced PDFs")
        md.extend([f"- {x}" for x in missing_pdfs[:50]])
        if len(missing_pdfs) > 50:
            md.append(f"- ... and {len(missing_pdfs) - 50} more")
    md.append("")

    md.append("## Connection Graph")
    md.append("### Edge type counts")
    for k, v in edge_type_counts.most_common():
        md.append(f"- **{k}**: {v}")
    md.append("")

    md.append("### Top degree nodes")
    for pid, dcount in top_degree:
        title = (node_by_id.get(pid, {}) or {}).get("title", "")
        suffix = f"- {title}" if title else ""
        md.append(f"- **{pid}**: degree={dcount} {suffix}")
    md.append("")

    md.append("### Top semantic edges (by similarity)")
    if sem_edges:
        for sim, s, t in sem_edges:
            md.append(f"- **{s} ↔ {t}**: sim={sim:.3f}")
    else:
        md.append("(No semantic edges found)")
    md.append("")

    md.append("### Mermaid preview (top semantic edges)")
    md.extend(mermaid)
    md.append("")

    md.append("## Synthesis Matrix")
    md.append("### Key metrics")
    md.append(f"- **Rows:** {len(synthesis_rows)}")
    md.append(f"- **Placeholder rows:** {pending_placeholders}")
    md.append("")

    md.append("## Notes Spot-Check (manual suggestion)")
    md.append("Open a few notes in:")
    md.append("`pipeline/notes/<bucket>/<tier>/<paper_id>.md`")
    md.append("")
    md.append("Suggested quick checks:")
    md.append("- Title/authors correct in header")
    md.append("- Key Findings are populated and readable")
    md.append("- Methods/Results sections not obviously truncated")
    md.append("")

    out_path = run_root / "REVIEW_REPORT.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to run folder (processed/runs/<run_name>)")
    parser.add_argument("--sources", default="sources", help="Sources folder")
    parser.add_argument("--priority", default="sources/PRIORITY_READING_LIST.csv", help="Priority list CSV")
    args = parser.parse_args()

    run_root = Path(args.run)
    sources_dir = Path(args.sources)
    priority_csv = Path(args.priority)

    out = build_review(run_root=run_root, sources_dir=sources_dir, priority_csv=priority_csv)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
