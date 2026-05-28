"""
Generate Hierarchical Confusion Matrices (Top Class & Subclass)

Top class별 confusion matrix와 각 top class 내 subclass별 confusion matrix를 생성합니다.

Usage:
    python generate_hierarchical_confusion_matrices.py --mode both
    python generate_hierarchical_confusion_matrices.py --mode audio
    python generate_hierarchical_confusion_matrices.py --mode all
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from losses import HierarchicalProxyLoss
from utils import get_subconfig, set_seed, build_class_to_topclass_mapping
from models import BaseClassifier
from dataset_utils import HATRDataset

# Paths
data_dir = get_subconfig("output_path")
processed_basename = get_subconfig("processed_dataset_csv")
class_dict_json = os.path.join(data_dir, get_subconfig("class_dict_json"))
top_class_dict_json = os.path.join(data_dir, get_subconfig("top_class_dict_json"))

# Identify main (10k) dataset key from config
datasets_cfg = get_subconfig("datasets")
all_dataset_names = list(datasets_cfg.keys())
main_dataset_name = None
for name in all_dataset_names:
    lname = name.lower()
    if '10k' in lname or 'bsd10k' in lname:
        main_dataset_name = name
        break

if main_dataset_name is None:
    raise ValueError("10k dataset not found in config!")

base = os.path.splitext(processed_basename)[0]
candidate_main = os.path.join(data_dir, main_dataset_name, f"{base}_{main_dataset_name}.csv")

if not os.path.exists(candidate_main):
    raise FileNotFoundError(f"10k dataset file not found: {candidate_main}")

prepared_dataset_main = candidate_main


def load_class_dict(json_path):
    """Load class dictionary from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_class_labels(class_dict):
    """Get sorted list of class labels."""
    return sorted(class_dict.keys(), key=lambda x: class_dict[x])


def invert_mapping(mapping):
    return {value: key for key, value in mapping.items()}


def predict_on_dataset(model, data_loader, device, criterion=None):
    """Predict labels for entire dataset using proxy-based classification."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            labels = data['class_idx'].to(device)
            audio_emb = data.get('audio_embedding', None)
            text_emb = data.get('text_embedding', None)

            if audio_emb is not None:
                audio_emb = audio_emb.to(device)
            if text_emb is not None:
                text_emb = text_emb.to(device)

            z, _, _ = model(audio_emb, text_emb)

            if criterion is not None:
                child_proxies = F.normalize(criterion.child_proxies, dim=1)
                child_logits = torch.matmul(z, child_proxies.T)
            else:
                child_logits = z

            _, predicted = torch.max(child_logits, 1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return np.array(all_preds), np.array(all_labels)


def compute_normalized_confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix normalized to 0-1 range (per class)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    return cm_normalized


def plot_confusion_matrix(cm_normalized, class_labels, title, save_path, figsize=(14, 12)):
    """Plot normalized confusion matrix."""
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Normalized Value (0-1)'},
        square=True,
        vmin=0,
        vmax=1
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def generate_top_class_confusion_matrices(mode, model_output_dir, class_dict, top_class_dict, k_folds=5):
    """Generate Top Class level confusion matrices."""
    print(f"\n{'='*80}")
    print(f"Generating Top Class Confusion Matrices: Mode={mode}")
    print(f"{'='*80}\n")
    
    class_id_to_name = invert_mapping(class_dict)
    top_class_id_to_name = invert_mapping(top_class_dict)
    class_ids = sorted(class_id_to_name.keys())
    top_class_ids = sorted(top_class_id_to_name.keys())
    top_class_labels = [top_class_id_to_name[top_class_id] for top_class_id in top_class_ids]
    
    class_to_topclass = build_class_to_topclass_mapping(class_dict, top_class_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load 10k dataset
    database = pd.read_csv(prepared_dataset_main)
    print(f"Loaded 10k dataset: {len(database)} samples")
    
    labels = database["class_idx"].tolist()
    seed = set_seed()
    
    # Create train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=2192, random_state=seed)
    _, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    test_df = database.iloc[test_idx].reset_index(drop=True)
    
    print(f"Test set size: {len(test_df)}\n")
    
    # Create test dataset
    test_dataset = HATRDataset(test_df, aug=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Set up output directory
    mode_output_dir = os.path.join(model_output_dir, mode)
    top_class_confusion_dir = os.path.join(mode_output_dir, "top_class_confusion_matrices")
    os.makedirs(top_class_confusion_dir, exist_ok=True)
    
    # Collect confusion matrices from all folds
    all_fold_cms = []
    all_preds_per_fold = []
    all_labels_per_fold = []
    
    for fold in range(k_folds):
        fold_dir = os.path.join(mode_output_dir, f"fold_{fold}")
        model_path = os.path.join(fold_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            print(f"⚠ Fold {fold}: Model not found at {model_path}")
            continue
        
        print(f"Processing Fold {fold}...")
        
        # Load model
        emb_size_audio = 512 if mode in ['audio', 'both'] else 0
        emb_size_text = 512 if mode in ['text', 'both'] else 0
        
        model = BaseClassifier(
            hidden_size=128,
            num_classes=len(class_dict),
            emb_size_audio=emb_size_audio,
            emb_size_text=emb_size_text,
            dropout=0.1,
            use_batch_norm=True,
            mode=mode
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])

        criterion = None
        if checkpoint.get('use_hierarchical_loss', False):
            criterion = HierarchicalProxyLoss(
                embedding_dim=128 // 2,
                num_parents=len(top_class_dict),
                num_children=len(class_dict),
            ).to(device)
            criterion.load_state_dict(checkpoint['criterion_state'])
            criterion.eval()

        # Predict on test set
        y_pred, y_true = predict_on_dataset(model, test_loader, device, criterion)

        # Convert to top class labels
        y_true_top = np.array([class_to_topclass[int(idx)] for idx in y_true])
        y_pred_top = np.array([class_to_topclass[int(idx)] for idx in y_pred])
        
        # Compute normalized confusion matrix for top classes
        cm_normalized = compute_normalized_confusion_matrix(
            y_true_top, y_pred_top, len(top_class_labels)
        )
        
        all_fold_cms.append(cm_normalized)
        all_preds_per_fold.append(y_pred_top)
        all_labels_per_fold.append(y_true_top)
        
        # Plot fold-specific confusion matrix
        fold_title = f"Top Class CM - Mode={mode} | Fold {fold}"
        fold_save_path = os.path.join(top_class_confusion_dir, f"fold_{fold}_top_class_cm.png")
        plot_confusion_matrix(cm_normalized, top_class_labels, fold_title, fold_save_path, figsize=(10, 9))
        
        print(f"  ✓ Fold {fold} top class confusion matrix saved\n")
    
    if not all_fold_cms:
        print(f"✗ No folds processed for mode {mode}")
        return
    
    # Compute average confusion matrix
    avg_cm = np.mean(all_fold_cms, axis=0)
    
    # Plot average
    avg_title = f"Average Top Class CM - Mode={mode} | Across {len(all_fold_cms)} Folds"
    avg_save_path = os.path.join(top_class_confusion_dir, "average_top_class_cm.png")
    plot_confusion_matrix(avg_cm, top_class_labels, avg_title, avg_save_path, figsize=(10, 9))
    
    print(f"✓ Average top class confusion matrix saved\n")
    
    # Save results
    all_preds_combined = np.concatenate(all_preds_per_fold)
    all_labels_combined = np.concatenate(all_labels_per_fold)
    
    results = {
        'mode': mode,
        'num_folds': len(all_fold_cms),
        'num_top_classes': len(top_class_labels),
        'top_class_labels': top_class_labels,
        'average_cm': avg_cm.tolist(),
        'overall_accuracy': np.mean(all_preds_combined == all_labels_combined),
    }
    
    results_path = os.path.join(top_class_confusion_dir, "top_class_cm_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Top class results saved\n")


def generate_subclass_confusion_matrices(mode, model_output_dir, class_dict, top_class_dict, k_folds=5):
    """Generate Sub-class level confusion matrices within each top class."""
    print(f"\n{'='*80}")
    print(f"Generating Sub-class Confusion Matrices: Mode={mode}")
    print(f"{'='*80}\n")
    
    class_id_to_name = invert_mapping(class_dict)
    top_class_id_to_name = invert_mapping(top_class_dict)
    class_ids = sorted(class_id_to_name.keys())
    top_class_ids = sorted(top_class_id_to_name.keys())
    top_class_labels = [top_class_id_to_name[top_class_id] for top_class_id in top_class_ids]
    
    class_to_topclass = build_class_to_topclass_mapping(class_dict, top_class_dict)
    
    # Build reverse mapping: top_class -> subclasses
    topclass_to_subclasses = defaultdict(list)
    for class_id, top_class_id in class_to_topclass.items():
        topclass_to_subclasses[top_class_id].append(class_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load 10k dataset
    database = pd.read_csv(prepared_dataset_main)
    labels = database["class_idx"].tolist()
    seed = set_seed()
    
    # Create train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=2192, random_state=seed)
    _, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    test_df = database.iloc[test_idx].reset_index(drop=True)
    
    print(f"Test set size: {len(test_df)}\n")
    
    # Create test dataset
    test_dataset = HATRDataset(test_df, aug=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Set up output directory
    mode_output_dir = os.path.join(model_output_dir, mode)
    subclass_confusion_dir = os.path.join(mode_output_dir, "subclass_confusion_matrices")
    os.makedirs(subclass_confusion_dir, exist_ok=True)
    
    # Collect predictions from all folds
    all_preds_per_fold = []
    all_labels_per_fold = []
    
    for fold in range(k_folds):
        fold_dir = os.path.join(mode_output_dir, f"fold_{fold}")
        model_path = os.path.join(fold_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            print(f"⚠ Fold {fold}: Model not found")
            continue
        
        print(f"Processing Fold {fold}...")
        
        # Load model
        emb_size_audio = 512 if mode in ['audio', 'both'] else 0
        emb_size_text = 512 if mode in ['text', 'both'] else 0
        
        model = BaseClassifier(
            hidden_size=128,
            num_classes=len(class_dict),
            emb_size_audio=emb_size_audio,
            emb_size_text=emb_size_text,
            dropout=0.1,
            use_batch_norm=True,
            mode=mode
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])

        criterion = None
        if checkpoint.get('use_hierarchical_loss', False):
            criterion = HierarchicalProxyLoss(
                embedding_dim=128 // 2,
                num_parents=len(top_class_dict),
                num_children=len(class_dict),
            ).to(device)
            criterion.load_state_dict(checkpoint['criterion_state'])
            criterion.eval()

        # Predict on test set
        y_pred, y_true = predict_on_dataset(model, test_loader, device, criterion)

        all_preds_per_fold.append(y_pred)
        all_labels_per_fold.append(y_true)
        
        print(f"  ✓ Fold {fold} predictions collected\n")
    
    if not all_preds_per_fold:
        print(f"✗ No folds processed for mode {mode}")
        return
    
    # Combine predictions from all folds
    all_preds_combined = np.concatenate(all_preds_per_fold)
    all_labels_combined = np.concatenate(all_labels_per_fold)
    
    # Generate confusion matrices for each top class
    print(f"\nGenerating confusion matrices for {len(top_class_labels)} top classes...\n")
    
    results = {'mode': mode, 'num_folds': len(all_preds_per_fold)}
    
    for top_class_label in top_class_labels:
        top_class_id = top_class_dict[top_class_label]
        subclasses = sorted(topclass_to_subclasses[top_class_id])
        
        if not subclasses:
            continue
        
        # Filter predictions/labels for this top class
        mask = np.array([class_to_topclass[int(idx)] == top_class_id 
                        for idx in all_labels_combined])
        
        y_true_filtered = all_labels_combined[mask]
        y_pred_filtered = all_preds_combined[mask]
        
        if len(y_true_filtered) == 0:
            print(f"  ⚠ {top_class_label}: No test samples")
            continue
        
        # Convert to subclass indices; map out-of-group predictions to Other
        subclass_to_idx = {subclass: i for i, subclass in enumerate(subclasses)}
        other_idx = len(subclasses)
        
        y_true_subclass = np.array([subclass_to_idx[int(idx)] for idx in y_true_filtered])
        y_pred_subclass = np.array([
            subclass_to_idx.get(int(idx), other_idx)
            for idx in y_pred_filtered
        ])
        
        # Compute normalized confusion matrix with an explicit Other column/row
        cm_normalized = compute_normalized_confusion_matrix(
            y_true_subclass, y_pred_subclass, len(subclasses) + 1
        )
        
        # Plot confusion matrix
        title = f"Sub-class CM - {top_class_label} | Mode={mode}"
        safe_top_class_label = str(top_class_label).replace(os.sep, "_")
        save_path = os.path.join(subclass_confusion_dir, f"{safe_top_class_label}_subclass_cm.png")
        
        figsize = (max(8, (len(subclasses) + 1) // 2), max(8, (len(subclasses) + 1) // 2))
        subclass_labels = [class_id_to_name[subclass_id] for subclass_id in subclasses] + ["Other"]
        plot_confusion_matrix(cm_normalized, subclass_labels, title, save_path, figsize=figsize)
        
        # Compute accuracy
        accuracy = np.mean(y_pred_subclass == y_true_subclass)
        
        results[top_class_label] = {
            'num_subclasses': len(subclasses),
            'num_samples': len(y_true_subclass),
            'accuracy': float(accuracy),
            'confusion_matrix': cm_normalized.tolist()
        }
        
        print(f"  ✓ {top_class_label} ({len(subclasses)} subclasses, {len(y_true_subclass)} samples, acc: {accuracy:.4f})")
    
    # Save results
    results_path = os.path.join(subclass_confusion_dir, "subclass_cm_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Sub-class confusion matrices saved\n")


def main():
    parser = argparse.ArgumentParser(description="Generate hierarchical confusion matrices")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "both", "audio"],
        help="Mode to process. 'all' generates for both modes."
    )
    parser.add_argument(
        "--output-dir",
        default="./model_output",
        help="Root output directory containing fold results"
    )
    args = parser.parse_args()
    
    modes = ['both', 'audio'] if args.mode == 'all' else [args.mode]
    
    # Load class dictionaries
    class_dict = load_class_dict(class_dict_json)
    top_class_dict = load_class_dict(top_class_dict_json)
    
    for mode in modes:
        generate_top_class_confusion_matrices(
            mode, args.output_dir, class_dict, top_class_dict, k_folds=5
        )
        generate_subclass_confusion_matrices(
            mode, args.output_dir, class_dict, top_class_dict, k_folds=5
        )
    
    print(f"\n{'='*80}")
    print("Hierarchical confusion matrix generation completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
