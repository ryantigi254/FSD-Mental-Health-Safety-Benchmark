# Qwen3-8B-MLX Study A cache review

Source: `Mac-setup/results/qwen3-8b-mlx/study_a_generations.jsonl`  
Run mode: generation-only cache (two modes per ID: CoT, Direct).

## Summary
- Entries: 1,820
- Distinct sample IDs: 300 (expected Study A set)
- Status counts: 600 ok, 1,220 error
- Per ID: all 300 IDs have at least one ok for both CoT and Direct (but many extra failed attempts logged)

## Mode-level counts
- CoT: 300 ok, 611 error (total 911 attempts)
- Direct: 300 ok, 609 error (total 909 attempts)

## Errors
- Top errors:
  - `Generation timed out after 60s`: 20
  - Multiple `Connection refused` to `http://localhost:1234/v1/chat/completions` (LM Studio not reachable)
- All error rows have empty `output_text`.

## Persona coverage
- `persona_id` is missing (`null`) on all cache rows, so persona-level verification against v2 personas cannot be performed from this run.

## Gaps / follow-ups
- Persona metadata is absent; regenerate with persona_id included to audit per-persona coverage.
- Connection/timeouts indicate LM Studio was unreachable or slow for many attempts; rerun after ensuring the server is up and responding.
- Current cache contains both ok and errored attempts; metrics-from-cache should only use `status=ok` rows.

