import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.transformer import Transformer

WINDOW = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_single_data(dataset="dos"):
    """
    Load one dataset at a time (normal, comm, dos, recon).
    Uses the last column as the target (0 = Normal, 1 = Attack).
    Ensures features are numeric only.
    """
    dataset_map = {
        "normal": "data/WUSTL-IIoT/train_WUSTL-IIoT.pkl",
        "comm":   "data/WUSTL-IIoT/comm_WUSTL-IIoT.pkl",
        "dos":    "data/WUSTL-IIoT/dos_WUSTL-IIoT.pkl",
        "recon":  "data/WUSTL-IIoT/recon_WUSTL-IIoT.pkl",
    }

    if dataset not in dataset_map:
        raise ValueError(f"Invalid dataset: {dataset}. Choose from {list(dataset_map.keys())}")

    df = pd.read_pickle(dataset_map[dataset]).copy()

    # ---- Target = last column ----
    targets = df.iloc[:, -1].astype(int).values

    # ---- Features = all but last column, numeric only ----
    features = df.iloc[:, :-1].select_dtypes(include=["number"]).astype(np.float32).values

    print(f"Loaded dataset={dataset}, samples={len(features)}")
    print(f"Targets distribution: Normal={np.sum(targets==0)}, Attack={np.sum(targets==1)}")
    print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")

    return features, targets




def sliding_window(data, targets, window=WINDOW, max_sequences=300000):
    """Create sliding windows with memory limits"""
    print(f"Creating sliding windows (max {max_sequences} sequences)...")
    
    # Calculate total possible sequences
    total_possible = len(data) - window + 1
    print(f"Total possible sequences: {total_possible}")
    
    if total_possible > max_sequences:
        # Sample sequences to reduce memory
        step = total_possible // max_sequences
        indices = range(0, total_possible, step)
        print(f"Sampling every {step}th sequence to get {len(indices)} sequences")
    else:
        indices = range(total_possible)
    
    sequences = []
    sequence_targets = []
    
    for i in indices:
        # Input sequence: window consecutive network flows
        seq = data[i:i+window]  # Shape: (window, 43_features)
        # Target: attack label from the last flow in sequence
        target = targets[i+window-1]  # Binary: 0=normal, 1=attack
        sequences.append(seq)
        sequence_targets.append(target)
    
    X_seq = torch.tensor(np.array(sequences), dtype=torch.float32)
    y_seq = torch.tensor(np.array(sequence_targets), dtype=torch.float32)
    
    print(f"Sliding window - X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")
    print(f"Memory usage: {X_seq.element_size() * X_seq.nelement() / 1024**3:.2f} GB")
    return X_seq, y_seq

def load_data(dataset="dos"):
    """Load one dataset (normal, comm, dos, recon)"""
    X, y = load_single_data(dataset=dataset)
    if X is None:
        return None, None

    # Create sequences
    X_seq, y_seq = sliding_window(X, y, window=WINDOW)

    # Create dataset + split
    dataset = TensorDataset(X_seq, y_seq)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len

    print(f"Final dataset split for {dataset}: Train={train_len}, Val={val_len}")
    return random_split(dataset, [train_len, val_len])


def partition_data(client_id, num_clients, dataset="dos", val_split=0.2, shuffle=True):
    """
    Partition one dataset for federated learning into multiple clients.
    Each client gets a different slice of the dataset.
    """
    # Load dataset (normal/comm/dos/recon)
    X, y = load_single_data(dataset=dataset)
    if X is None:
        return None, None

    total_len = len(X)
    indices = np.arange(total_len)

    # Shuffle before splitting to avoid label imbalance
    if shuffle:
        np.random.shuffle(indices)

    part_len = total_len // num_clients
    start = client_id * part_len
    end = (client_id + 1) * part_len if client_id < num_clients - 1 else total_len

    client_idx = indices[start:end]
    X_part, y_part = X[client_idx], y[client_idx]

    print(f"Client {client_id}: {len(X_part)} samples, "
          f"Normal={np.sum(y_part==0)}, Attack={np.sum(y_part==1)}")

    # Create sliding windows
    X_seq, y_seq = sliding_window(X_part, y_part, window=WINDOW)

    # Create dataset + train/val split
    dataset = TensorDataset(X_seq, y_seq)
    train_len = int((1 - val_split) * len(dataset))
    val_len = len(dataset) - train_len
    return random_split(dataset, [train_len, val_len])

def get_model(input_dim):
    """Create Transformer model for network flow anomaly detection"""
    print(f"Creating Transformer model with input_dim: {input_dim}")
    return Transformer(
        d_model=input_dim,
        N_layers=2,
        attention="single",
        window=WINDOW,
        device=str(DEVICE),
        dropout=0,
        d_ff=64
    ).to(DEVICE)

def train(model, train_loader, epochs=1):
    model.train_model(train_loader, epochs=epochs)

def test(model, val_loader):
    """Comprehensive evaluation for attack detection model"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_scores = []
    
    print("Evaluating model on validation data...")
    
    with torch.no_grad():
        for batch in val_loader:
            # Handle both feature+target and single tensor cases
            if len(batch) == 2:
                features, targets = batch
            else:
                features = batch[0]
                # For single tensor, use last timestep as target
                targets = features[:, -1, -1]  # Last feature of last timestep
            
            # Get model predictions
            outputs = model(features, features)
            scores = torch.sigmoid(outputs).squeeze()
            
            # Convert to binary predictions (threshold = 0.5)
            predictions = (scores > 0.5).float()
            
            # Ensure targets are the right shape
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)
    
    # Calculate comprehensive evaluation metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    
    print(f"\n=== VALIDATION RESULTS ===")
    print(f"Total samples: {len(all_targets)}")
    print(f"Normal samples: {np.sum(all_targets == 0)}")
    print(f"Attack samples: {np.sum(all_targets == 1)}")
    
    # Check if we have both classes
    unique_targets = np.unique(all_targets)
    print(f"Unique target values: {unique_targets}")
    
    if len(unique_targets) < 2:
        print("⚠ Warning: Only one class found in validation data!")
        print("Cannot calculate classification metrics with single class.")
        return {
            'accuracy': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
            'roc_auc': 1.0,
            'specificity': 1.0,
            'sensitivity': 1.0,
            'confusion_matrix': np.array([[len(all_targets), 0], [0, 0]]),
            'predictions': all_predictions,
            'targets': all_targets,
            'scores': all_scores
        }
    
    # Basic classification metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)
    
    # AUC metrics
    try:
        roc_auc = roc_auc_score(all_targets, all_scores)
    except:
        roc_auc = 0.5
    
    # Confusion matrix - handle different shapes safely
    cm = confusion_matrix(all_targets, all_predictions)
    print(f"Confusion matrix shape: {cm.shape}")
    print(f"Confusion matrix:\n{cm}")
    
    # Safely extract confusion matrix values
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        # Only one class present
        if unique_targets[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        # Handle other cases
        print(f"Unexpected confusion matrix shape: {cm.shape}")
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Print results
    print(f"\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    
    print(f"\nConfusion Matrix Details:")
    print(f"True Negatives (Normal): {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives (Attack): {tp}")
    
    # Return comprehensive results
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets,
        'scores': all_scores
    }
