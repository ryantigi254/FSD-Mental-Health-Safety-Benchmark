
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
from pathlib import Path
import re
import sys

def patch_bias_notebook(notebook):
    """Patch Study A Bias notebook to include CI visualization."""
    print("Patching Study A Bias Analysis Notebook...")
    
    # Patch 1: Update Metrics Loading to include CIs
    for cell in notebook.cells:
        if cell.cell_type == "code" and "rows.append" in cell.source:
            # Check if likely the right cell
            if "Silent Bias Rate" in cell.source and "ci_low" not in cell.source:
                print("  - Patching metrics loading loop...")
                # We simply replace the 'rows.append({' block with one that includes CIs
                # Using regex or simple string replacement if robust enough
                # The target block is:
                #             "Silent Bias Rate": metrics.get("silent_bias_rate", 0.0),
                #             "Refusal Rate": metrics.get("refusal_rate", 0.0),
                
                replacement = """            "Silent Bias Rate": metrics.get("silent_bias_rate", 0.0),
            "Silent Bias Rate CI Low": metrics.get("silent_bias_rate_ci_low", 0.0),
            "Silent Bias Rate CI High": metrics.get("silent_bias_rate_ci_high", 0.0),
            "Refusal Rate": metrics.get("refusal_rate", 0.0),"""
                
                cell.source = cell.source.replace(
                    '"Silent Bias Rate": metrics.get("silent_bias_rate", 0.0),',
                    replacement
                )
    
    # Patch 2: Update Plotting to show Error Bars
    for cell in notebook.cells:
        if cell.cell_type == "code" and "sns.barplot" in cell.source:
             if '"Silent Bias Rate"' in cell.source and "plt.errorbar" not in cell.source:
                print("  - Patching plotting code to include Error Bars...")
                # Completely replace the plotting cell
                new_source = """
# Plot Silent Bias Rate with Confidence Intervals
plt.figure(figsize=(12, 8))

# Sort by Silent Bias Rate
df = df.sort_values("Silent Bias Rate", ascending=True)

# Calculate error bars (asymmetric)
# matplotlb errorbar takes shape (2, N) -> [[lower_errors], [upper_errors]]
# lower_error = val - low
# upper_error = high - val
y_err = [
    (df["Silent Bias Rate"] - df["Silent Bias Rate CI Low"]).clip(lower=0),
    (df["Silent Bias Rate CI High"] - df["Silent Bias Rate"]).clip(lower=0)
]

# Create horizontal bar plot using matplotlib directly for easier error bar handling
bars = plt.barh(df["Model"], df["Silent Bias Rate"], xerr=y_err, capsize=5, color=sns.color_palette("viridis", len(df)))

plt.title("Silent Bias Rate with 95% Confidence Intervals (Lower is Better)")
plt.xlabel("Proportion of Demographic-Aligned Errors")
plt.axvline(0.2, color="r", linestyle="--", label="Safety Threshold (0.2)")

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
             va='center', fontsize=9)

plt.legend()
plt.tight_layout()
plt.show()
"""
                cell.source = new_source
                
    return notebook

def convert_notebook(notebook_path, output_dir):
    print(f"Processing {notebook_path.name}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    # Apply patches if needed
    if notebook_path.name == "study_a_bias_analysis.ipynb":
        nb = patch_bias_notebook(nb)
    
    # Execute
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
    except Exception as e:
        print(f"Error executing {notebook_path.name}: {e}")
        # Continue to export even if errors (to see partial results)
    
    # Export to HTML
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = True # Optional: hide code cells
    (body, resources) = html_exporter.from_notebook_node(nb)
    
    output_path = output_dir / f"{notebook_path.stem}.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(body)
    
    print(f"  -> Saved to {output_path}")

def main():
    base_dir = Path(__file__).parent.parent.parent
    notebooks_dir = base_dir / "notebooks"
    output_dir = notebooks_dir / "html"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    notebooks = [
        notebooks_dir / "study_a_analysis.ipynb",
        notebooks_dir / "study_a_bias_analysis.ipynb",
        notebooks_dir / "study_b_analysis.ipynb",
        notebooks_dir / "study_c_analysis.ipynb"
    ]
    
    for nb_path in notebooks:
        if nb_path.exists():
            convert_notebook(nb_path, output_dir)
        else:
            print(f"Warning: Notebook not found: {nb_path}")

if __name__ == "__main__":
    main()
