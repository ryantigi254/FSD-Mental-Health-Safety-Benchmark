================================================================================
RAW vs CLEANED COMPARISON - ALL STUDIES
================================================================================


============================================================
STUDY_A
============================================================

Model                       Entries     Raw KB   Clean KB  Removed  ExtDiff
------------------------------------------------------------------------
deepseek-r1-lmstudio            300      683.2      681.1     0.3%       75
gpt-oss-20b                     300      350.4      349.7     0.2%        7
piaget-8b-local                 300      944.2      904.5     4.2%      117
psych-qwen-32b-local            300       44.2       44.1     0.0%        2
psyche-r1-local                 300       10.2       10.2     0.0%        1
psyllm-gml-local                300      919.8      917.4     0.3%       70
qwen3-lmstudio                  300      857.5      854.4     0.4%      121
qwq                             300     1095.5     1091.2     0.4%      127
------------------------------------------------------------------------
TOTAL                          2400     4904.9     4852.6     1.1%      520

Extraction difference rate: 21.7%
❌ HIGH IMPACT: Cleaning significantly affects extractions

============================================================
STUDY_B
============================================================

Model                       Entries     Raw KB   Clean KB  Removed  ExtDiff
------------------------------------------------------------------------
deepseek-r1-lmstudio            277     1010.4     1006.9     0.3%        0
gpt-oss-20b                     277     1545.6     1535.9     0.6%        0
piaget-8b-local                 277     1159.7     1054.0     9.1%        0
psych-qwen-32b-local            277      529.9      528.3     0.3%        0
psyche-r1-local                 277      267.5      267.3     0.1%        0
psyllm-gml-local                277      593.5      592.1     0.2%        0
qwen3-lmstudio                  277     1202.6     1190.4     1.0%        0
qwq                             277     1442.2     1428.4     1.0%        0
------------------------------------------------------------------------
TOTAL                          2216     7751.2     7603.2     1.9%        0

Extraction difference rate: 0.0%
✅ SAFE: Cleaning has minimal impact on extractions

============================================================
STUDY_C
============================================================

Model                       Entries     Raw KB   Clean KB  Removed  ExtDiff
------------------------------------------------------------------------
deepseek-r1-lmstudio              1        6.0        5.9     1.1%        0
gpt-oss-20b                       1        4.7        4.7     0.2%        0
psych-qwen-32b-local              1        1.2        1.2     0.1%        0
psyche-r1-local                   1        0.7        0.7     0.0%        0
psyllm-gml-local                  1        1.8        1.7     0.7%        0
qwen3-lmstudio                    1        5.5        5.3     3.1%        0
qwq                               1        5.3        5.1     2.8%        0
------------------------------------------------------------------------
TOTAL                             7       25.1       24.7     1.6%        0

Extraction difference rate: 0.0%
✅ SAFE: Cleaning has minimal impact on extractions

================================================================================
OVERALL SUMMARY
================================================================================

study_a: 2400 entries, 1.1% chars removed, 21.7% extraction diff ❌
study_b: 2216 entries, 1.9% chars removed, 0.0% extraction diff ✅
study_c: 7 entries, 1.6% chars removed, 0.0% extraction diff ✅
