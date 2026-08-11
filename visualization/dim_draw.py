import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap
from sklearn.preprocessing import StandardScaler

CELLLINE = 'K562'
RESOLUTION  = '25K'

for i in range(1, 23):
    feature_file = f'data/{CELLLINE}/{RESOLUTION}/feats/chr{i}_features'
    label_file = f'result/{CELLLINE}/{RESOLUTION}/predictions/chr{i}.txt'

    # Load data
    features = np.loadtxt(feature_file)  # Sample features
    labels_pred = np.loadtxt(label_file, skiprows=1)  # Labels and predictions

    # Split ground-truth labels and predictions
    labels = labels_pred[:, 0]  # First column: ground-truth labels
    predictions = labels_pred[:, 1]  # Second column: predictions

    # Normalize (Z-score standardization)
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)  # Standardized features

    # # t-SNE dimensionality reduction
    # tsne = TSNE(n_components=2, random_state=42)
    # features_2d = tsne.fit_transform(features)


    # # Isomap dimensionality reduction
    # isomap = Isomap(n_components=2)
    # features_2d = isomap.fit_transform(features)

    # UMAP dimensionality reduction
    umap_reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.5)
    features_2d = umap_reducer.fit_transform(features)

    # # PCA dimensionality reduction
    # pca = PCA(n_components=2)
    # features_2d = pca.fit_transform(features)

    print('Drawing...')
    size = 8
    color_map = {0: 'blue', 1: 'red'}

    # Plot ground-truth labels and save
    plt.figure(figsize=(8, 8))
    for label in np.unique(labels):
        plt.scatter(features_2d[labels == label, 0], features_2d[labels == label, 1],
                    c=color_map[label], label=f"Label {int(label)}", alpha=0.6, s=size)
    plt.title("t-SNE Visualization (True Labels)")
    plt.legend()
    plt.savefig(f"draws/groundtruths_chr{i}.pdf", dpi=300)
    plt.close()

    # # Plot predictions and save
    # plt.figure(figsize=(6, 6))
    # for pred in np.unique(predictions):
    #     plt.scatter(features_2d[predictions == pred, 0], features_2d[predictions == pred, 1],
    #                 c=color_map[pred], label=f"Prediction {int(pred)}", alpha=0.6, s=size)
    # plt.title("t-SNE Visualization (Predictions)")
    # plt.legend()
    # plt.savefig("predictions.png", dpi=300)
    # plt.close()

    print("Visualizations saved.")
