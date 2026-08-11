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
CELLLINE = 'MOLM13'
RESOLUTION  = '25K'
GCL_LR = 0.01
GCL_EPOCHS = 10  # Set to 0 to load the saved model directly
PRE_LR = 0.01
PRE_EPOCHS = 150

HiC_THRESHOLD = 2  # Number of edges / graph density
GAUSSIAN_SIGMA = 0.1
DROP_RATIO = 0.8

HIDDEN_DIM = 64
EMBEDDING_DIM = 14
PRE_HIDDEN_DIM = 64

ALPHA = 0.6
lambda1 = 0.4

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

# Feature augmentation
def add_gaussian_noise(features, sigma=0.1):
    noise = torch.randn_like(features) * sigma
    return features + noise

# Topology augmentation
def edge_dropout_aug(edge_index, drop_ratio):
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=edge_index.device) > drop_ratio
    return edge_index[:, mask]

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

def compute_mse_adj_rec(H, original_adj, block_size=1000):
    """
    Compute the MSE loss for adjacency matrix reconstruction in blocks to reduce GPU memory usage.
    H: tensor of shape [num_nodes, hidden_dim], on GPU
    original_adj: tensor of shape [num_nodes, num_nodes], on GPU
    Returns: MSE loss (torch.Tensor)
    """
    num_nodes = H.size(0)
    mse_sum = 0.0

    similarity_max = torch.max(torch.matmul(H, H.t()))
    for i in range(0, num_nodes, block_size):
        end_i = min(i + block_size, num_nodes)
        H_block = H[i:end_i]  # [block, hidden_dim]
        similarity = torch.matmul(H_block, H.t())  # [block, num_nodes]
        similarity = similarity / similarity_max  # Normalize

        # Compute MSE
        mse_block = F.mse_loss(similarity, original_adj[i:end_i], reduction='sum')
        mse_sum += mse_block

        # Free GPU cache
        del H_block, similarity
        torch.cuda.empty_cache()

    mse = mse_sum / (num_nodes * num_nodes)  # Mean MSE
    return mse

# Feature correlation between augmented views
def compute_feature_correlation(H1, H2):
    """
    Compute feature similarity and return the mean cosine similarity.
    H1, H2: tensors of shape [num_nodes, hidden_dim], on GPU
    Returns: mean feature similarity (float)
    """
    feature_corr = F.cosine_similarity(H1, H2, dim=1)  # [num_nodes]
    feature_corr_cpu = feature_corr.cpu()
    identity = torch.ones_like(feature_corr_cpu)
    mse = F.mse_loss(feature_corr_cpu, identity, reduction='sum').item()
    mse = mse / feature_corr_cpu.size(0)
    return mse

# GCL pre-training
input_dim = data_list[0].x.size(1)
model = GCNEncoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, out_dim=EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=GCL_LR)

best_val_loss = float('inf')
best_model = None

for epoch in range(1, GCL_EPOCHS + 1):
    model.train()
    total_loss = 0
    for data in data_list:
        optimizer.zero_grad()
        # Original features and adjacency matrix
        data = data.to(device)  # Move the whole Data object to the device
        original_x = data.x  # [num_nodes, num_features]
        original_edge_index = data.edge_index  # [2, num_edges]
        num_nodes = original_x.size(0)

        # Build the original adjacency matrix
        original_adj = torch.zeros(num_nodes, num_nodes, device=device)
        original_adj[original_edge_index[0], original_edge_index[1]] = 1

        # Embeddings H from the GCN encoder
        H = model(original_x, original_edge_index)  # [num_nodes, EMBEDDING_DIM], on GPU

        # Augmentation 1: Gaussian noise on features
        aug1_x = add_gaussian_noise(original_x, sigma=GAUSSIAN_SIGMA)

        # Augmentation 2: random edge dropout
        aug2_edge_index = edge_dropout_aug(original_edge_index, drop_ratio=DROP_RATIO)

        # Embeddings H1 and H2 for the augmented views
        H1 = model(aug1_x, original_edge_index)  # [num_nodes, EMBEDDING_DIM], on GPU
        H2 = model(original_x, aug2_edge_index)  # [num_nodes, EMBEDDING_DIM], on GPU

        # Loss1: feature reconstruction and topology correlation
        mse_features = None
        mse_adj = compute_mse_adj_rec(H, original_adj)  # Block-wise MSE, returns a float
        if ALPHA != 0:
            mse_features = F.mse_loss(original_x, H)  # Kept on GPU
            loss1 = ALPHA * mse_features + (1 - ALPHA) * mse_adj  # loss1 includes the mse_adj term, which is already on CPU
        else:
            loss1 = mse_adj

        # Loss3: feature similarity between augmented views
        loss3 = compute_feature_correlation(H1, H2)  # Mean cosine similarity, returns a float

        # Total loss
        loss = loss1 * lambda1 + loss3 * (1-lambda1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Free GPU cache
        del original_x, original_edge_index, original_adj, H, aug1_x, aug2_edge_index, H1, H2, mse_features, mse_adj
        torch.cuda.empty_cache()

    avg_train_loss = total_loss / len(data_list)

    print(f'GCL_Epoch {epoch}: Train Loss={avg_train_loss:.4f}')

# Save the best model (when retraining)
if GCL_EPOCHS != 0:
    best_model = copy.deepcopy(model.state_dict())
    torch.save(best_model, os.path.join(best_model_path, f'best_{CELLLINE}_{RESOLUTION}_GCL.pkl'))
    print(f'Best GCL_model saved with Val Loss={best_val_loss:.4f}')

# 22-fold cross-validation: chromosome i is held out for testing, chromosome (i+1)%22 for validation, the rest for training
for test_idx in range(22):
    print(f'----------------------------------------')
    print(f'   Test_chr:{test_idx+1}')
    print(f'----------------------------------------')
    val_idx = (test_idx + 1) % 22  # Next chromosome serves as validation
    train_idx = [j for j in range(22) if j != test_idx and j != val_idx]
    train_set = [data_list[j] for j in train_idx]
    val_set = data_list[val_idx]
    test_set = data_list[test_idx]

    # Load the best GCL model
    model.load_state_dict(torch.load(os.path.join(best_model_path, f'best_{CELLLINE}_{RESOLUTION}_GCL.pkl')))
    model.eval()  # Switch to evaluation mode

    # Collect training-set embeddings and labels
    train_embeddings = []
    train_labels = []
    for data in train_set:
        with torch.no_grad():
            H = model(data.x.to(device), data.edge_index.to(device))
            train_embeddings.append(H.cpu())
            train_labels.append(data.y.cpu())
        # Free GPU cache
        del H
        torch.cuda.empty_cache()
    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Collect validation-set embeddings and labels
    with torch.no_grad():
        H_val = model(val_set.x.to(device), val_set.edge_index.to(device)).cpu()
        val_embeddings = H_val
        val_labels = val_set.y.cpu()
        # Free GPU cache
        del H_val
        torch.cuda.empty_cache()

    # Collect test-set embeddings and labels
    with torch.no_grad():
        H_test = model(test_set.x.to(device), test_set.edge_index.to(device)).cpu()
        test_embeddings = H_test
        test_labels = test_set.y.cpu()
        # Free GPU cache
        del H_test
        torch.cuda.empty_cache()
        
    # Classifier head
    classifier = LogReg(ft_in=(EMBEDDING_DIM), hidden_dim=PRE_HIDDEN_DIM, nb_classes=1).to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=PRE_LR)
    classification_loss_fn = nn.BCEWithLogitsLoss()  # Instantiate the loss function

    best_classifier_val_loss = float('inf')
    best_classifier_val_auprc = 0.0
    best_classifier_state = None
    
    # Move data to device
    train_emb = train_embeddings.to(device)
    train_lbl = train_labels.to(device).float()  # BCEWithLogitsLoss expects float labels
    val_emb = val_embeddings.to(device)
    val_lbl = val_labels.to(device).float()
    test_emb = test_embeddings.to(device)
    test_lbl = test_labels.to(device).float()

    no_improve_epochs = 0
    early_stop_patience = 5  # Stop early after 5 epochs without improvement

    # Train the classifier
    for epoch in range(1, PRE_EPOCHS +1):
        classifier.train()
        classifier_optimizer.zero_grad()
        outputs = classifier(train_emb).squeeze()  # [num_nodes]
        loss = classification_loss_fn(outputs, train_lbl)
        loss.backward()
        classifier_optimizer.step()
        train_loss = loss.item()

        # Validation
        classifier.eval()
        with torch.no_grad():

            # Training-set metrics
            train_probs = torch.sigmoid(outputs).cpu().numpy()
            train_pred = (train_probs >= 0.5).astype(int)
            train_labels_np = train_lbl.cpu().numpy()
            acc_train, auroc_train, auprc_train, f1_train, precision_train, recall_train = compute_metrics(train_labels_np, train_pred, train_probs)

            # Validation-set metrics
            outputs_val = classifier(val_emb).squeeze()
            val_loss = classification_loss_fn(outputs_val, val_lbl).item()
            val_probs = torch.sigmoid(outputs_val).cpu().numpy()
            val_pred = (val_probs >= 0.5).astype(int)
            val_labels_np = val_lbl.cpu().numpy()
            acc_val, auroc_val, auprc_val, f1_val, precision_val, recall_val = compute_metrics(val_labels_np, val_pred, val_probs)
            
            # Test-set metrics (for monitoring only)
            logits_test = classifier(test_emb).squeeze()
            probs_test = torch.sigmoid(logits_test).cpu().numpy()  # Probabilities
            preds_test = (probs_test >= 0.5).astype(int)  # Binary classification threshold
            labels_test = test_lbl.cpu().numpy()
            acc, auroc, auprc, f1, precision, recall = compute_metrics(labels_test, preds_test, probs_test)
            
        # Print losses and metrics
        print(f'Fold {test_idx + 1}. Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        print(f'Train Metrics: ACC={acc_train:.4f}, AUROC={auroc_train:.4f}, '
            f'AUPRC={auprc_train:.4f}, F1={f1_train:.4f}, PRECISION={precision_train:.4f}, '
            f'RECALL={recall_train:.4f}')
        print(f'Val Metrics: ACC={acc_val:.4f}, AUROC={auroc_val:.4f}, '
            f'AUPRC={auprc_val:.4f}, F1={f1_val:.4f}, PRECISION={precision_val:.4f}, '
            f'RECALL={recall_val:.4f}')
        print(f'Test Metrics: ACC={acc:.4f}, AUROC={auroc:.4f}, '
            f'AUPRC={auprc:.4f}, F1={f1:.4f}, PRECISION={precision:.4f}, '
            f'RECALL={recall:.4f}\n')
        
        # if val_loss < best_classifier_val_loss:
        #     best_classifier_val_loss = val_loss
        #     best_classifier_state = copy.deepcopy(classifier.state_dict())
        if auprc_val > best_classifier_val_auprc:
            best_classifier_val_auprc = auprc_val
            best_classifier_state = copy.deepcopy(classifier.state_dict())
            
    # Save the best classifier of this fold
    torch.save(best_classifier_state, os.path.join(best_model_path, f'best_{CELLLINE}_{RESOLUTION}_fold{test_idx+1}_PRE.pkl'))
    print(f'Best Classifier for fold {test_idx+1} saved with Val Loss={best_classifier_val_loss:.4f}')

    # ------------------- Evaluation -------------------
    # Load the best classifier
    classifier.load_state_dict(torch.load(os.path.join(best_model_path, f'best_{CELLLINE}_{RESOLUTION}_fold{test_idx+1}_PRE.pkl')))
    classifier.eval()

    # Evaluate on the test set
    with torch.no_grad():
        logits_test = classifier(test_emb).squeeze()
        probs_test = torch.sigmoid(logits_test).cpu().numpy()  # Probabilities
        preds_test = (probs_test >= 0.5).astype(int)  # Binary classification threshold
        labels_test = test_lbl.cpu().numpy()

    # Compute evaluation metrics
    acc, auroc, auprc, f1, precision, recall = compute_metrics(labels_test, preds_test, probs_test)

    # Append results to a TXT file
    result_file = f'result/train_{CELLLINE}_{RESOLUTION}_Ress.txt'
    with open(result_file, 'a') as f:
        f.write(f'{test_idx+1}\t{acc:.4f}\t{auroc:.4f}\t{auprc:.4f}\t{f1:.4f}\t{precision:.4f}\t{recall:.4f}\n')
    print(f'Resolution{RESOLUTION}, Chr{test_idx +1}: Test Metrics: ACC={acc:.4f}, AUROC={auroc:.4f}, AUPRC={auprc:.4f}, F1={f1:.4f}, PRECISION={precision:.4f}, RECALL={recall:.4f}\n')

    # Free GPU memory
    del train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl, classifier
    torch.cuda.empty_cache()

print("Training and evaluation finished. Test metrics for all folds saved to", result_file)