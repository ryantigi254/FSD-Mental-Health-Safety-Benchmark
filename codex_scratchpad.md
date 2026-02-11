### 2026-02-08 21:05:26 GMT
- Goal:
  - Commit and push ignore protection for local Uni-setup Conda folders.
- Actions taken:
  - Committed `.gitignore` update on branch `study-b-pressure-scaling`.
  - Pushed commit and set upstream tracking to `origin/study-b-pressure-scaling`.
- What changed:
  - Remote branch now contains ignore rules preventing accidental commits of `.conda-envs` and `.conda-pkgs`.

### 2026-02-08 20:53:32 GMT
- Goal:
  - Keep local Conda env/package folders out of Git for Uni-setup.
- Actions taken:
  - Added `.gitignore` rules for `Assignment 2/reliable_clinical_benchmark/Uni-setup/.conda-envs/` and `.conda-pkgs/`.
  - Verified ignore matching with `git check-ignore -v`.
- What changed:
  - These folders are now protected from accidental commit/push.

