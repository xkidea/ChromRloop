import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import StandardScaler

CELLLINE = 'MOLM13'
RESOLUTION = '25K'
METHOD = 'RF'

# Accumulate data for all chromosomes
all_features = []
all_labels = []
all_predictions = []
all_chr_indices = []

# Read data for all 22 chromosomes
for i in range(1, 23):
    feature_file = f'data/{CELLLINE}/{RESOLUTION}/feats/chr{i}_features'
    # label_file = f'result/{CELLLINE}/{RESOLUTION}/predictions/chr{i}.txt'
    label_file = f'result/{CELLLINE}/{RESOLUTION}/{METHOD}_predictions/chr{i}.txt'
    
    # Load data
    features = np.loadtxt(feature_file)  # Sample features
    labels_pred = np.loadtxt(label_file, skiprows=1)  # Labels and predictions

    # Split ground-truth labels and predictions
    labels = labels_pred[:, 0]  # First column: ground-truth labels
    predictions = labels_pred[:, 1]  # Second column: predictions

    # Record the chromosome index of each sample
    chr_indices = np.full(len(labels), i)  # One chromosome index per sample

    # Store into the global dataset
    all_features.append(features)
    all_labels.append(labels)
    all_predictions.append(predictions)
    all_chr_indices.append(chr_indices)

# Concatenate all chromosomes
all_features = np.vstack(all_features)
all_labels = np.hstack(all_labels)
all_predictions = np.hstack(all_predictions)
all_chr_indices = np.hstack(all_chr_indices)  # Chromosome index per sample

# Normalize (Z-score standardization)
scaler = StandardScaler()
features_normalized = scaler.fit_transform(all_features)  # Standardize all features

# UMAP dimensionality reduction
umap_reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.5)
features_2d = umap_reducer.fit_transform(features_normalized)

print('Drawing...')
size = 1  # Point size
color_map = {0: 'blue', 1: 'red'}

# # 2D view of ground-truth labels
# plt.figure(figsize=(10, 8))
# for label in np.unique(all_labels):
#     plt.scatter(features_2d[all_labels == label, 0], features_2d[all_labels == label, 1],
#                 c=color_map[label], label=f"Label {int(label)}", alpha=0.6, s=size)
# plt.title("UMAP Visualization (True Labels)")
# plt.legend()
# plt.savefig(f"draws_{CELLLINE}/groundtruths_all.pdf", dpi=300)
# plt.savefig(f"draws_{CELLLINE}/groundtruths_all.png", dpi=300)
# plt.close()

# 2D view of predictions
plt.figure(figsize=(10, 8))
for pred in np.unique(all_predictions):
    plt.scatter(features_2d[all_predictions == pred, 0], features_2d[all_predictions == pred, 1],
                c=color_map[pred], label=f"Prediction {int(pred)}", alpha=0.6, s=size)
plt.title("UMAP Visualization (Predictions)")
plt.legend()
plt.savefig(f"draws_{CELLLINE}/predictions_all_{METHOD}.pdf", dpi=300)
plt.savefig(f"draws_{CELLLINE}/predictions_all_{METHOD}.png", dpi=300)
plt.close()

print("Visualizations for all chromosomes saved.")
