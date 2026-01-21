from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class RepetitionHit:
    file_path: Path
    line_num: int
    reason: str
    snippet: str


def iter_texts(jsonl_path: Path) -> Iterable[tuple[int, str]]:
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("response_text") or obj.get("output_text") or ""
            if text:
                yield line_num, text


def has_consecutive_line_repeats(text: str, min_len: int = 30, min_repeats: int = 3) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < min_repeats:
        return False
    count = 1
    prev = lines[0]
    for line in lines[1:]:
        if line == prev and len(line) >= min_len:
            count += 1
            if count >= min_repeats:
                return True
        else:
            prev = line
            count = 1
    return False


def has_repeated_sentence_ngrams(text: str, n: int = 2, min_sent_len: int = 20, min_hits: int = 2) -> bool:
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    if len(sentences) < n * 2:
        return False
    ngrams = []
    for i in range(len(sentences) - n + 1):
        gram = " ".join(sentences[i : i + n]).strip()
        if len(gram) >= min_sent_len:
            ngrams.append(gram)
    counts = Counter(ngrams)
    return any(v >= min_hits for v in counts.values())


def scan_results(results_dir: Path) -> list[RepetitionHit]:
    hits: list[RepetitionHit] = []
    for jsonl_path in sorted(results_dir.glob("**/*_generations.jsonl")):
        for line_num, text in iter_texts(jsonl_path):
            if has_consecutive_line_repeats(text):
                snippet = " ".join(text.splitlines()[:3])[:200]
                hits.append(
                    RepetitionHit(jsonl_path, line_num, "consecutive_line_repeat", snippet)
                )
                continue
            if has_repeated_sentence_ngrams(text, n=2):
                snippet = " ".join(SENT_SPLIT.split(text)[:3])[:200]
                hits.append(
                    RepetitionHit(jsonl_path, line_num, "repeated_sentence_bigram", snippet)
                )
    return hits


def main() -> int:
    results_dir = Path("results")
    hits = scan_results(results_dir)
    if not hits:
        print("No obvious repetition found with strict exact-match checks.")
        return 0

    print(f"Found {len(hits)} potential repetition hits:")
    for hit in hits[:20]:
        rel = hit.file_path.as_posix()
        print(f"- {rel} (line {hit.line_num}): {hit.reason}")
        print(f"  {hit.snippet}")
    if len(hits) > 20:
        print(f"... and {len(hits) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
