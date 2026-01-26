================================================================================
CLEANING COMPARISON REPORT: Raw vs Original Cleaned vs Optimized Cleaned
================================================================================

SUMMARY TABLE
--------------------------------------------------------------------------------
Model                      Entries Orig Diffs  Opt Diffs  Opt Better?
--------------------------------------------------------------------------------
deepseek-r1-lmstudio           600   315 (52.5%)     0 ( 0.0%)          YES
gpt-oss-20b                    600    82 (13.7%)     0 ( 0.0%)          YES
piaget-8b-local                600   253 (42.2%)    21 ( 3.5%)          YES
psych-qwen-32b-local           600   153 (25.5%)     0 ( 0.0%)          YES
psyche-r1-local                600   154 (25.7%)     0 ( 0.0%)          YES
psyllm-gml-local               600   349 (58.2%)     0 ( 0.0%)          YES
qwen3-lmstudio                 600   258 (43.0%)     0 ( 0.0%)          YES
qwq                            600   239 (39.8%)     0 ( 0.0%)          YES
--------------------------------------------------------------------------------
TOTAL                         4800  1803 (37.6%)    21 ( 0.4%)

================================================================================
SIZE COMPARISON (Characters)
--------------------------------------------------------------------------------
Model                               Raw   Orig Clean    Opt Clean  Opt Savings
--------------------------------------------------------------------------------
deepseek-r1-lmstudio          2,125,448    2,132,659    2,125,448        0 (0.0%)
gpt-oss-20b                   2,605,016    2,588,354    2,605,016        0 (0.0%)
piaget-8b-local               2,521,601    2,456,896    2,469,417   52,184 (2.1%)
psych-qwen-32b-local          1,358,473    1,353,735    1,358,473        0 (0.0%)
psyche-r1-local                 265,238      273,795      265,093      145 (0.1%)
psyllm-gml-local              1,679,258    1,693,019    1,679,258        0 (0.0%)
qwen3-lmstudio                2,932,728    2,916,614    2,932,728        0 (0.0%)
qwq                           3,087,654    3,082,935    3,087,654        0 (0.0%)
--------------------------------------------------------------------------------
TOTAL                        16,575,416   16,498,007   16,523,087

================================================================================
CONCLUSION
================================================================================

OPTIMIZED CLEANING IS BETTER:
  - Original cleaning caused 1803 diagnosis extraction differences (37.6%)
  - Optimized cleaning caused 21 diagnosis extraction differences (0.4%)
  - Improvement: 1782 fewer differences

RECOMMENDATION:
  Optimized cleaning has minimal impact on diagnosis extraction (<5% difference).
  SAFE to use cleaned data for metrics calculation.
