import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
RESULTS_DIR = Path("metric-results")
OUTPUT_DIR = Path("Assignment 2/docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_study_a():
    data = []
    path = RESULTS_DIR / "study_a"
    if not path.exists(): return pd.DataFrame()
    for file in path.glob("*_metrics.json"):
        if file.name == "all_models_metrics.json": continue
        with open(file, "r") as f:
            d = json.load(f)
            d["model"] = file.name.replace("_metrics.json", "")
            data.append(d)
    return pd.DataFrame(data)

def load_study_b():
    path = RESULTS_DIR / "study_b" / "sycophancy_metrics.json"
    if path.exists():
        with open(path, "r") as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()

def load_study_c():
    path = RESULTS_DIR / "study_c" / "drift_metrics.json"
    if path.exists():
        with open(path, "r") as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()

# Study A Plots
df_a = load_study_a()
if not df_a.empty:
    # Faithfulness Gap
    plt.figure()
    df_sorted = df_a.sort_values("faithfulness_gap", ascending=False)
    colors = ["green" if x > 0.1 else "orange" for x in df_sorted["faithfulness_gap"]]
    sns.barplot(data=df_sorted, x="model", y="faithfulness_gap", palette=colors)
    plt.axhline(0.1, color="red", linestyle="--", label="Threshold (0.1)")
    plt.xticks(rotation=45, ha="right")
    plt.title("Faithfulness Gap by Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "study_a_faithfulness_gap.png")
    plt.close()

# Study B Plots
df_b = load_study_b()
if not df_b.empty:
    # Sycophancy Prob
    plt.figure()
    df_sorted = df_b.sort_values("sycophancy_probability", ascending=True)
    colors = ["green" if x < 0.2 else "red" for x in df_sorted["sycophancy_probability"]]
    sns.barplot(data=df_sorted, x="model", y="sycophancy_probability", palette=colors)
    plt.axhline(0.2, color="red", linestyle="--", label="Threshold (0.2)")
    plt.xticks(rotation=45, ha="right")
    plt.title("Sycophancy Probability (Lower is Better)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "study_b_sycophancy_prob.png")
    plt.close()
    
    # Sycophancy vs Hallucination (Zoomed)
    plt.figure()
    if "evidence_hallucination" in df_b.columns:
        plt.scatter(df_b["sycophancy_probability"], df_b["evidence_hallucination"], s=100)
        for i, row in df_b.iterrows():
            plt.annotate(row["model"], (row["sycophancy_probability"], row["evidence_hallucination"]))
        
        # Zoom logic
        x_min, x_max = df_b["sycophancy_probability"].min(), df_b["sycophancy_probability"].max()
        padding = (x_max - x_min) * 0.2 if x_max != x_min else 0.05
        plt.xlim(x_min - padding, x_max + padding)
        
        plt.xlabel("Sycophancy Probability")
        plt.ylabel("Evidence Hallucination")
        plt.title("Sycophancy vs Hallucination (Focused)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "study_b_syc_vs_hav.png")
        plt.close()

# Study C Plots
df_c = load_study_c()
if not df_c.empty:
    # Check column names
    recall_col = "recall_at_10"
    if recall_col not in df_c.columns:
        # Fallback search
        for c in df_c.columns:
            if "recall" in c and "10" in c:
                recall_col = c
                break
    
    if recall_col in df_c.columns:
        plt.figure()
        df_sorted = df_c.sort_values(recall_col, ascending=False)
        sns.barplot(data=df_sorted, x="model", y=recall_col, palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.title("Entity Recall at Turn 10")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "study_c_recall_t10.png")
        plt.close()

print("Plots generated in", OUTPUT_DIR)

