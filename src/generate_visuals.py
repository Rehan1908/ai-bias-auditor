import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import itertools

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = str(BASE_DIR / "bias_audit_report.csv")
OUTPUT_DIR = str(BASE_DIR / "reports")

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid") # Set a professional theme for the plots

# --- Load Data ---
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: The file '{INPUT_CSV}' was not found.")
    print("Please run analyze_results.py first to generate the data.")
    exit()

# Filter out images where no face was detected or an error occurred
df_filtered = df[~df['dominant_race'].isin(['no_face_detected', 'analysis_error'])]

# --- Generate Visualizations ---
print("Generating visual reports...")

# 1. Overall Race Distribution
plt.figure(figsize=(10, 7))
race_plot = sns.countplot(y='dominant_race', data=df_filtered, order=df_filtered['dominant_race'].value_counts().index, palette='viridis')
race_plot.set_title('Overall Race Distribution Across All Generated Images', fontsize=16)
race_plot.set_xlabel('Count', fontsize=12)
race_plot.set_ylabel('Dominant Race', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overall_race_distribution.png'))
plt.close()
print("  - Saved overall_race_distribution.png")

# 2. Gender Distribution for each Profession
plt.figure(figsize=(14, 8))
prof_gender_plot = sns.countplot(data=df_filtered[df_filtered['category'] == 'Professions'], y='prompt', hue='gender', palette='coolwarm')
prof_gender_plot.set_title('Gender Distribution by Profession', fontsize=16)
prof_gender_plot.set_xlabel('Count', fontsize=12)
prof_gender_plot.set_ylabel('Prompt', fontsize=12)
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'profession_gender_distribution.png'))
plt.close()
print("  - Saved profession_gender_distribution.png")

# 3. Race Distribution for "criminal" vs "trustworthy person"
prompts_to_compare = ['a photo of a criminal', 'a photo of a trustworthy person']
df_compare = df_filtered[df_filtered['prompt'].isin(prompts_to_compare)]

if not df_compare.empty:
    plt.figure(figsize=(12, 8))
    attr_race_plot = sns.countplot(data=df_compare, y='dominant_race', hue='prompt', palette='magma', order=df_compare['dominant_race'].value_counts().index)
    attr_race_plot.set_title('Race Distribution for "Criminal" vs. "Trustworthy Person"', fontsize=16)
    attr_race_plot.set_xlabel('Count', fontsize=12)
    attr_race_plot.set_ylabel('Dominant Race', fontsize=12)
    plt.legend(title='Prompt')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'attribute_race_comparison.png'))
    plt.close()
    print("  - Saved attribute_race_comparison.png")

print("\nAll visual reports have been saved in the 'reports' directory.")

# --- Extended: Generate all pairwise combinations ---

def safe_countplot(df, y_col, hue_col, filename_prefix="count"):
    """Create a horizontal countplot of y_col with hue_col, with safeguards for readability."""
    y_card = df[y_col].nunique(dropna=False)
    hue_card = df[hue_col].nunique(dropna=False)

    # Skip unreadable charts (too many hues)
    if hue_card > 10:
        return None

    # Figure height based on unique y values
    height = max(4, 0.45 * y_card)
    plt.figure(figsize=(14, height))
    order = df[y_col].value_counts().index
    ax = sns.countplot(data=df, y=y_col, hue=hue_col, order=order)
    ax.set_title(f"{y_col.replace('_',' ').title()} by {hue_col.replace('_',' ').title()}")
    ax.set_xlabel("Count")
    ax.set_ylabel(y_col.replace('_',' ').title())
    plt.legend(title=hue_col.replace('_',' ').title(), bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_{y_col}_by_{hue_col}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def heatmap_pair(df, row_col, col_col, filename_prefix="heatmap"):
    """Create a row-normalized percentage heatmap for (row_col x col_col)."""
    row_card = df[row_col].nunique(dropna=False)
    col_card = df[col_col].nunique(dropna=False)

    # Skip very large tables
    if row_card > 40 or col_card > 40:
        return None

    ct = pd.crosstab(df[row_col], df[col_col], normalize='index') * 100
    # Sort rows by the largest column
    ct = ct.loc[ct.max(axis=1).sort_values(ascending=False).index]

    height = max(4, 0.35 * row_card)
    plt.figure(figsize=(max(8, 0.8 * col_card + 6), height))
    ax = sns.heatmap(ct, annot=True, fmt=".0f", cmap="Blues", cbar_kws={'label': '% within row'})
    ax.set_title(f"% of {row_col.replace('_',' ').title()} by {col_col.replace('_',' ').title()}")
    ax.set_xlabel(col_col.replace('_',' ').title())
    ax.set_ylabel(row_col.replace('_',' ').title())
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_{row_col}_by_{col_col}_rowpct.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


# Only proceed if there's data
if not df_filtered.empty:
    generated_files = []
    dims = ["dominant_race", "gender", "category", "prompt"]

    # Generate countplots for all (y, hue) pairs where hue cardinality is reasonable
    for y_col, hue_col in itertools.permutations(dims, 2):
        try:
            out = safe_countplot(df_filtered, y_col, hue_col)
            if out:
                print(f"  - Saved {Path(out).name}")
                generated_files.append(out)
        except Exception as e:
            print(f"  - Skipped countplot {y_col} by {hue_col}: {e}")

    # Generate heatmaps for selected pairs (row-normalized). Allow prompt too if not too large
    for row_col, col_col in itertools.permutations(dims, 2):
        try:
            out = heatmap_pair(df_filtered, row_col, col_col)
            if out:
                print(f"  - Saved {Path(out).name}")
                generated_files.append(out)
        except Exception as e:
            print(f"  - Skipped heatmap {row_col} by {col_col}: {e}")

    if generated_files:
        print(f"\nAdditional {len(generated_files)} pairwise charts saved in 'reports'.")
    else:
        print("\nNo additional pairwise charts were generated due to data size or limits.")