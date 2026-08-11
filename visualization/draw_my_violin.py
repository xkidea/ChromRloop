import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Folder paths
METHODS = ["ours", "GAT", "GCN", "MLP", "RF", "LINE"]

CELLLINE = 'MOLM13'
folder_path = f"result/{CELLLINE}/25K"

# Accumulate scores across methods
data = []

# Read predictions for each method
for method in METHODS:
    method_folder = os.path.join(folder_path, f"{method}_predictions")

    # Iterate over all files in the folder
    for filename in os.listdir(method_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(method_folder, filename)
            # Read file contents
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    # Expected format: label, predicted label, predicted probability
                    label, predicted, prob = line.strip().split()  # Whitespace-separated columns
                    data.append([method, int(label), float(prob)])  # Score = predicted probability

# Build a DataFrame
df = pd.DataFrame(data, columns=["Model", "Label", "Score"])

# Plot style without background grid
sns.set_theme(style="white")  # Plain white background for a cleaner look

# Create the figure
plt.figure(figsize=(10, 8))

# Violin plot for label 0
plt.subplot(1, 2, 1)  # 1 row, 2 columns: first panel
sns.violinplot(
    y="Model",
    x="Score",
    data=df[df["Label"] == 0],  # Label-0 samples only
    inner=None,  # Density shape only
    palette="Blues_r",
    bw_adjust=0.7
)

# Overlay quartile lines and median markers
for i, method in enumerate(METHODS):
    subset = df[(df["Model"] == method) & (df["Label"] == 0)]["Score"]
    q1 = subset.quantile(0.25)  # 25th percentile
    median = subset.median()    # Median
    q3 = subset.quantile(0.75)  # 75th percentile

    # Quartile range (blue)
    plt.gca().hlines(y=i, xmin=q1, xmax=q3, color="#1f77b4", linewidth=2, alpha=0.8)

    # Median marker (orange with white edge)
    plt.gca().scatter(median, i, color="#ff7f0e", edgecolors="white", s=60, linewidth=1.5, zorder=3)

plt.xlabel("Confidence score", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.title("Confidence Distribution (Label = 0)", fontsize=14)

# Violin plot for label 1
plt.subplot(1, 2, 2)  # 1 row, 2 columns: second panel
sns.violinplot(
    y="Model",
    x="Score",
    data=df[df["Label"] == 1],  # Label-1 samples only
    inner=None,  # Density shape only
    palette="Reds_r",
    bw_adjust=0.7
)

# Overlay quartile lines and median markers
for i, method in enumerate(METHODS):
    subset = df[(df["Model"] == method) & (df["Label"] == 1)]["Score"]
    q1 = subset.quantile(0.25)  # 25th percentile
    median = subset.median()    # Median
    q3 = subset.quantile(0.75)  # 75th percentile

    # Quartile range (orange)
    plt.gca().hlines(y=i, xmin=q1, xmax=q3, color="#ff7f0e", linewidth=2, alpha=0.8)

    # Median marker (blue with white edge)
    plt.gca().scatter(median, i, color="#1f77b4", edgecolors="white", s=60, linewidth=1.5, zorder=3)

plt.xlabel("Confidence score", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.title("Confidence Distribution (Label = 1)", fontsize=14)

# Adjust the layout
plt.tight_layout()

# Save the figure
plt.savefig(f"{CELLLINE}_violins.pdf", dpi=300, bbox_inches='tight')
plt.close()

print("Violin plots saved.")
