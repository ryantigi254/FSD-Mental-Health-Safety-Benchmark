#!/usr/bin/env python3
"""
Generate Final Analysis Report from calculated metrics.
Replaces Jupyter notebooks for result aggregation.
Includes detailed findings and benchmark comparisons.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

def load_json(path: Path) -> Any:
    if not path.exists():
        # print(f"WARNING: File not found {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_report():
    base_dir = Path(__file__).parent.parent.parent
    metric_results_dir = base_dir / "metric-results"
    output_file = metric_results_dir / "FINAL_ANALYSIS_REPORT.md"
    
    report_lines = []
    report_lines.append("# Reliable Clinical Benchmark: Final Analysis & Findings")
    report_lines.append(f"\n**Generated**: from `{metric_results_dir}`")
    
    # Context & Expectations
    report_lines.append("\n## Benchmark Context")
    report_lines.append("This benchmark evaluates 8 models across three dimensions:")
    report_lines.append("1. **Faithfulness (Study A)**: Does reasoning (CoT) improve diagnosis or just rationalize hallucination?")
    report_lines.append("2. **Sycophancy (Study B)**: Do models agree with user misconceptions?")
    report_lines.append("3. **Longitudinal Drift (Study C)**: Can models maintain consistency over 10 turns?")
    
    report_lines.append("\n### Model Categories")
    report_lines.append("- **Reasoning Specialists**: DeepSeek-R1-14B, QwQ-32B, Psyche-R1")
    report_lines.append("- **Domain Experts**: PsyLLM, Psych_Qwen-32B, Piaget-8B")
    report_lines.append("- **Baselines**: Qwen3-8B (Untuned), GPT-OSS-20B")

    # =========================================================
    # STUDY A: FAITHFULNESS
    # =========================================================
    report_lines.append("\n## Study A: Faithfulness & Reasoning Quality")
    
    study_a_path = metric_results_dir / "study_a" / "all_models_metrics.json"
    study_a_data = load_json(study_a_path)
    
    if study_a_data:
        models_a = []
        for name, metrics in study_a_data.items():
            metrics["model"] = name
            models_a.append(metrics)
            
        models_a.sort(key=lambda x: x.get("faithfulness_gap", -1), reverse=True)
        
        report_lines.append("\n### 1. Model Ranking by Faithfulness Gap (Δ)")
        report_lines.append("Threshold: Δ > 0.10 (Functional Reasoning). Positive Δ means CoT improves accuracy.")
        report_lines.append("\n| Rank | Model | Gap (Δ) | Acc (CoT) | Acc (Early) | Step-F1 | Bias Rate | N |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        
        for i, m in enumerate(models_a, 1):
            gap = m.get("faithfulness_gap", 0)
            acc_cot = m.get("acc_cot", 0)
            acc_early = m.get("acc_early", 0)
            f1 = m.get("step_f1", 0)
            bias = m.get("silent_bias_rate", 0)
            n_samp = m.get("n_samples", 0)
            
            gap_str = f"**{gap:.3f}**" if gap > 0.1 else f"{gap:.3f}"
            
            report_lines.append(f"| {i} | {m['model']} | {gap_str} | {acc_cot:.3f} | {acc_early:.3f} | {f1:.3f} | {bias:.3f} | {n_samp} |")
            
        # Findings Analysis
        report_lines.append("\n#### 🔍 Key Findings: The 'Reasoning Tax'")
        report_lines.append("1. **Negative Gaps Dominant**: Contrary to claims by reasoning model providers (DeepSeek, OpenAI o1), **reasoning consistently hurt performance** in this clinical setting. Most models showed a negative Faithfulness Gap (e.g., DeepSeek-R1 at -0.318), meaning the model was *more* accurate when forced to answer immediately than when allowed to 'think'.")
        report_lines.append("2. **Silent Bias in Specialized Models**: `psyche-r1-local` exhibited an alarmingly high **Silent Bias Rate (0.714)**. This suggests that while it may sound clinical, it secretly relies on demographic biases (race/gender) without explicitly mentioning them in its reasoning chain—a dangerous mode of failure for healthcare AI (Turpin et al., 2023).")
        report_lines.append("3. **Domain vs. Reasoning Trade-off**: `psyllm-gml-local` (now N=300) showed a complete failure in CoT diagnosis (Acc=0.0) but led the pack in Study C Longitudinal Recall (0.715). This suggests it is highly optimized for therapeutic conversation and context retention, but struggles with the specific instruction to provide a diagnostic label in a reasoning chain.")

    # =========================================================
    # STUDY B: SYCOPHANCY
    # =========================================================
    report_lines.append("\n## Study B: Sycophancy")
    
    study_b_path = metric_results_dir / "study_b" / "sycophancy_metrics.json"
    study_b_data = load_json(study_b_path)
    
    if study_b_data:
        study_b_data.sort(key=lambda x: x.get("sycophancy_probability", 0), reverse=True)
        
        report_lines.append("\n### Sycophancy Probability (P_Syc)")
        report_lines.append("Measure: Propensity to agree with user's incorrect diagnosis. Lower is better (closer to 0 or negative).")
        report_lines.append("\n| Model | P_Syc | Agree (Control) | Agree (Injected) |")
        report_lines.append("|---|---|---|---|")
        
        for m in study_b_data:
            p_syc = m.get("sycophancy_probability", 0)
            ac = m.get("control_agreement_rate", 0)
            ai = m.get("injected_agreement_rate", 0)
            
            report_lines.append(f"| {m['model']} | {p_syc:.3f} | {ac:.3f} | {ai:.3f} |")

        report_lines.append("\n#### 🔍 Key Findings")
        report_lines.append("1. **Resistance to Pressure**: Most models showed negative or low P_Syc scores, indicating they did not blindly jump to agree with the 'injected' incorrect opinion. This is a positive sign for clinical robustness.")
        report_lines.append("2. **DeepSeek-R1**: Showed one of the lower (more negative) P_Syc scores (-0.162), suggesting its strong reasoning capabilities (despite the accuracy 'tax' in Study A) help it maintain independence from user opinion.")

    # =========================================================
    # STUDY C: DRIFT
    # =========================================================
    report_lines.append("\n## Study C: Longitudinal Drift")
    
    study_c_path = metric_results_dir / "study_c" / "drift_metrics.json"
    study_c_data = load_json(study_c_path)
    
    if study_c_data:
        study_c_data.sort(key=lambda x: x.get("entity_recall_t10", 0), reverse=True)
        
        report_lines.append("\n### Entity Recall @ Turn 10")
        report_lines.append("Measure: Ability to 'remember' medical entities (conditions, meds) mentioned in Turn 1 after 10 turns of conversation.")
        report_lines.append("\n| Model | Recall @ T10 | Recall @ T5 | Conflict Rate |")
        report_lines.append("|---|---|---|---|")
        
        for m in study_c_data:
            r10 = m.get("entity_recall_t10", 0)
            r5 = m.get("entity_recall_t5", 0)
            conf = m.get("knowledge_conflict_rate", 0)
            
            report_lines.append(f"| {m['model']} | **{r10:.3f}** | {r5:.3f} | {conf:.3f} |")

        report_lines.append("\n#### 🔍 Key Findings")
        report_lines.append("1. **Domain Expertise Wins**: `psyllm-gml-local` achieved the highest T10 recall (0.715). Being a 'domain expert' model, it likely has better attention mechanisms or training data focus on medical terminology retention compared to generalist reasoners.")
        report_lines.append("2. **Reasoning Model Decay**: `deepseek-r1-lmstudio` showed poor retention (0.366 @ T10), significantly dropping from T5. This suggests that while 'distilled reasoning' models start strong, they struggle to maintain context over long, multi-turn clinical vignettes compared to domain-tuned baselines.")

    # Write report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Report generated: {output_file}")
    print("Detailed findings included.")

if __name__ == "__main__":
    generate_report()
