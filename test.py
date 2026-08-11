import os
import os.path as osp
import numpy as np
import pandas as pd
import copy
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import (roc_auc_score, confusion_matrix, precision_recall_curve, 
                             average_precision_score, accuracy_score, f1_score, recall_score, precision_score)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import glob

# Hyperparameters
CELLLINE = 'K562'
RESOLUTION  = '25K'

HiC_THRESHOLD = 1  # Number of edges / graph density

HIDDEN_DIM = 64
EMBEDDING_DIM = 12
PRE_HIDDEN_DIM = 64

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best_model_path = 'result/model/'

data_list = []
scaler = StandardScaler()

for i in range(1, 23):
    # Build file paths
    edge_file = glob.glob(f'data/{CELLLINE}/{RESOLUTION}/HiC/chr{i}_*')[0]
    feature_file = f'data/{CELLLINE}/{RESOLUTION}/feats/chr{i}_features'
    label_file = f'data/{CELLLINE}/{RESOLUTION}/Label/chr{i}_label.txt'
    if not os.path.exists(edge_file):
        raise FileNotFoundError(f"Edge file {edge_file} not found.")
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file {feature_file} not found.")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file {label_file} not found.")

    # Load the edge file
    edges_df = pd.read_csv(edge_file, sep='\t', header=None, names=['u', 'v', 'edge_val'])
    # Keep contacts above the HiC threshold
    edges_df = edges_df[edges_df['edge_val'] > HiC_THRESHOLD].reset_index(drop=True)
    # Convert to zero-based indexing
    edge_index = torch.tensor([edges_df['u'].values - 1, edges_df['v'].values - 1], dtype=torch.long)
    # Undirected graph: add reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)
    # Edge attributes
    edge_attr = torch.tensor(edges_df['edge_val'].values, dtype=torch.float).unsqueeze(1)  # Shape: [num_edges, 1]

    # Load the node features
    features_df = pd.read_csv(feature_file, sep='\t', header=None)
    features_scaled = scaler.fit_transform(features_df.values)  # Standardize each chromosome independently
    x = torch.tensor(features_scaled, dtype=torch.float)  # Shape: [num_nodes, num_features]

    # Load the node labels
    labels_df = pd.read_csv(label_file, sep='\t', header=None)
    y = torch.tensor(labels_df.values.squeeze(), dtype=torch.long)  # Shape: [num_nodes]

    # Create a Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    data_list.append(data)

print(f"Loaded data for {len(data_list)} chromosomes.")

# GCN encoder
class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.outlina = nn.Linear(input_dim+hidden_dim+out_dim, out_dim)
    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        # x = F.relu(x)
        x2 = self.conv2(x1, edge_index)
        h = torch.cat([x, x1, x2], dim=1)
        h = self.outlina(h)
        return h

# Classifier head
class LogReg(nn.Module):
    def __init__(self, ft_in, hidden_dim, nb_classes=1):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(ft_in, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, nb_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Evaluation metrics
def compute_metrics(y_true, y_pred, y_prob):
    """
    y_true: Ground truth labels (numpy array)
    y_pred: Predicted labels (numpy array)
    y_prob: Predicted probabilities for the positive class (numpy array)
    """
    acc = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float('nan')  # Only one class present
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = float('nan')

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, auroc, auprc, f1, precision, recall

# Load the pre-trained encoder
input_dim = data_list[0].x.size(1)
model = GCNEncoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, out_dim=EMBEDDING_DIM).to(device)
model.load_state_dict(torch.load(os.path.join(best_model_path, f'best_{CELLLINE}_{RESOLUTION}_GCL.pkl')))
model.eval()

# 22-fold cross-validation: chromosome i is held out for testing, chromosome (i+1)%22 for validation, the rest for training
for test_idx in range(22):
    print(f'----------------------------------------')
    print(f'   Test_chr:{test_idx+1}')
    print(f'----------------------------------------')

    used_idx = test_idx  # Or set an explicit fold index (base 0)
    
    test_set = data_list[test_idx]
    
    # Collect test-set embeddings and labels
    with torch.no_grad():
        H_test = model(test_set.x.to(device), test_set.edge_index.to(device)).cpu()
        test_embeddings = H_test
        test_labels = test_set.y.cpu()
        # Free GPU cache
        del H_test
        torch.cuda.empty_cache()
        
    # Load the classifier
    classifier = LogReg(ft_in=(EMBEDDING_DIM), hidden_dim=PRE_HIDDEN_DIM, nb_classes=1).to(device)
    classifier.load_state_dict(torch.load(os.path.join(best_model_path, f'best_{CELLLINE}_{RESOLUTION}_fold{used_idx+1}_PRE.pkl')))
    classifier.eval()

    # Move data to device
    test_emb = test_embeddings.to(device)
    test_lbl = test_labels.to(device).float()

    # Evaluate
    with torch.no_grad():
        logits_test = classifier(test_emb).squeeze()
        probs_test = torch.sigmoid(logits_test).cpu().numpy()  # Probabilities
        preds_test = (probs_test >= 0.5).astype(int)  # Binary classification threshold
        labels_test = test_lbl.cpu().numpy()

    # Compute evaluation metrics
    acc, auroc, auprc, f1, precision, recall = compute_metrics(labels_test, preds_test, probs_test)

    # Append results to a TXT file
    result_file = f'result/test_{CELLLINE}_{RESOLUTION}_Ress.txt'
    with open(result_file, 'a') as f:
        f.write(f'{test_idx+1}\t{acc:.4f}\t{auroc:.4f}\t{auprc:.4f}\t{f1:.4f}\t{precision:.4f}\t{recall:.4f}\n')
    print(f'Resolution{RESOLUTION}, Chr{test_idx +1}: Test Metrics: ACC={acc:.4f}, AUROC={auroc:.4f}, AUPRC={auprc:.4f}, F1={f1:.4f}, PRECISION={precision:.4f}, RECALL={recall:.4f}\n')

