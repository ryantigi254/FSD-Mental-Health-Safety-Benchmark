"""Extract and compare latency stats from vLLM benchmark vs HF-local JSONL files."""

import json
import sys
from pathlib import Path


def extract_stats(jsonl_path: Path):
    """Return dict with count, latency list, sum, mean, min, max."""
    entries = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    latencies = []
    for e in entries:
        meta = e.get("meta", {})
        lat = meta.get("latency_ms")
        if lat is not None:
            latencies.append(lat)

    if not latencies:
        return {"count": len(entries), "latencies": [], "sum": 0, "mean": 0, "min": 0, "max": 0}

    return {
        "count": len(entries),
        "latencies": latencies,
        "sum": sum(latencies),
        "mean": sum(latencies) / len(latencies),
        "min": min(latencies),
        "max": max(latencies),
    }


def main():
    uni_root = Path(__file__).resolve().parents[2]
    results = uni_root / "results"

    # vLLM benchmark (maxseq_4)
    vllm_path = results / "psyllm-gml-local" / "misc" / "vllm_benchmark" / "temp_generations" / "maxseq_4" / "study_a_generations_temp.jsonl"
    # HF-local prior run
    hflocal_path = results / "psyllm-gml-local" / "study_a_generations.jsonl"

    print("=" * 70)
    print("vLLM vs HF-Local Latency Comparison (PsyLLM-8B)")
    print("=" * 70)

    for label, path in [("vLLM (maxseq=4)", vllm_path), ("HF-local", hflocal_path)]:
        if not path.exists():
            print(f"\n{label}: FILE NOT FOUND ({path})")
            continue
        stats = extract_stats(path)
        print(f"\n{label} ({path.name}):")
        print(f"  Entries: {stats['count']}")
        print(f"  Entries with latency: {len(stats['latencies'])}")
        if stats["latencies"]:
            print(f"  Latency sum:  {stats['sum']:.0f} ms")
            print(f"  Latency mean: {stats['mean']:.1f} ms")
            print(f"  Latency min:  {stats['min']:.0f} ms")
            print(f"  Latency max:  {stats['max']:.0f} ms")

            # Per-mode breakdown
            modes = {}
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    mode = e.get("mode", "unknown")
                    lat = e.get("meta", {}).get("latency_ms")
                    if lat is not None:
                        modes.setdefault(mode, []).append(lat)

            for mode, lats in sorted(modes.items()):
                avg = sum(lats) / len(lats)
                print(f"  [{mode}] n={len(lats)} mean={avg:.1f}ms min={min(lats):.0f}ms max={max(lats):.0f}ms")
        else:
            print("  No latency data found in meta")

    # Also check other models for HF-local baselines
    print("\n" + "=" * 70)
    print("HF-Local Baselines (all models, study_a)")
    print("=" * 70)
    for model_dir in ["psyllm-gml-local", "piaget-8b-local", "psyche-r1-local", "psych-qwen-32b-local"]:
        p = results / model_dir / "study_a_generations.jsonl"
        if not p.exists():
            print(f"\n{model_dir}: NOT FOUND")
            continue
        stats = extract_stats(p)
        print(f"\n{model_dir}:")
        print(f"  Entries: {stats['count']}, with latency: {len(stats['latencies'])}")
        if stats["latencies"]:
            print(f"  Latency mean: {stats['mean']:.1f} ms ({stats['mean']/1000:.1f}s)")
            print(f"  Latency min/max: {stats['min']:.0f} / {stats['max']:.0f} ms")

            # First 6 entries only (same as benchmark sample size)
            first_6_lats = stats["latencies"][:6]
            if first_6_lats:
                avg6 = sum(first_6_lats) / len(first_6_lats)
                print(f"  First-6 latency mean: {avg6:.1f} ms ({avg6/1000:.1f}s) — comparable to 3-sample benchmark")

    # Raw JSON dump for report writing
    print("\n" + "=" * 70)
    print("RAW JSON for report")
    print("=" * 70)
    report_data = {}
    for label, path in [("vllm_maxseq4", vllm_path), ("hflocal", hflocal_path)]:
        if path.exists():
            stats = extract_stats(path)
            report_data[label] = {k: v for k, v in stats.items() if k != "latencies"}
            report_data[label]["first_6_mean"] = (
                sum(stats["latencies"][:6]) / min(6, len(stats["latencies"]))
                if stats["latencies"] else 0
            )
    print(json.dumps(report_data, indent=2))


if __name__ == "__main__":
    main()
