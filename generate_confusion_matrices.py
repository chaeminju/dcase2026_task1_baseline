"""
Generate Confusion Matrices for test set predictions

각 폴드별 confusion matrix 이미지와 전체 평균 confusion matrix를 생성합니다.
모든 값은 0-1 사이의 정규화된 값입니다.

Usage:
    python generate_confusion_matrices.py --mode both
    python generate_confusion_matrices.py --mode audio
    python generate_confusion_matrices.py --mode all
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
from utils import get_subconfig, set_seed
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
    """
    Compute confusion matrix normalized to 0-1 range (per class).
    
    각 행(실제 클래스)별로 정규화하여, 각 행의 합이 1이 되도록 함.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # 행별 정규화 (각 행의 합 = 1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # NaN 처리 (해당 클래스가 test set에 없는 경우)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    return cm_normalized


def plot_confusion_matrix(cm_normalized, class_labels, title, save_path, figsize=(14, 12)):
    """
    Plot normalized confusion matrix.
    
    Args:
        cm_normalized: 정규화된 confusion matrix (0-1)
        class_labels: 클래스 이름 리스트
        title: 그림 제목
        save_path: 저장 경로
        figsize: 그림 크기
    """
    plt.figure(figsize=figsize)
    
    # seaborn heatmap
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


def generate_confusion_matrices_for_mode(mode, model_output_dir, k_folds=5):
    """
    Generate confusion matrices for all folds and average.
    
    Args:
        mode: 'both' or 'audio'
        model_output_dir: Model output root directory
        k_folds: Number of folds
    """
    print(f"\n{'='*80}")
    print(f"Generating Confusion Matrices: Mode={mode}")
    print(f"{'='*80}\n")
    
    # Load class dictionaries
    class_dict = load_class_dict(class_dict_json)
    top_class_dict = load_class_dict(top_class_dict_json)
    class_labels = get_class_labels(class_dict)
    num_classes = len(class_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load 10k dataset
    database = pd.read_csv(prepared_dataset_main)
    print(f"Loaded 10k dataset: {len(database)} samples")
    
    labels = database["class_idx"].tolist()
    seed = set_seed()
    
    # Create train/test split (same as finetune)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=2192, random_state=seed)
    _, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    test_df = database.iloc[test_idx].reset_index(drop=True)
    
    print(f"Test set size: {len(test_df)}\n")
    
    # Create test dataset (no augmentation)
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
    confusion_dir = os.path.join(mode_output_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)
    
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
            num_classes=num_classes,
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
                num_children=num_classes,
            ).to(device)
            criterion.load_state_dict(checkpoint['criterion_state'])
            criterion.eval()

        # Predict on test set
        y_pred, y_true = predict_on_dataset(model, test_loader, device, criterion)
        
        # Compute normalized confusion matrix
        cm_normalized = compute_normalized_confusion_matrix(y_true, y_pred, num_classes)
        
        # Store for averaging
        all_fold_cms.append(cm_normalized)
        all_preds_per_fold.append(y_pred)
        all_labels_per_fold.append(y_true)
        
        # Plot fold-specific confusion matrix
        fold_title = f"Confusion Matrix - Mode={mode} | Fold {fold}"
        fold_save_path = os.path.join(confusion_dir, f"fold_{fold}_confusion_matrix.png")
        plot_confusion_matrix(cm_normalized, class_labels, fold_title, fold_save_path)
        
        print(f"  ✓ Fold {fold} confusion matrix saved\n")
    
    if not all_fold_cms:
        print(f"✗ No folds processed for mode {mode}")
        return
    
    # Compute average confusion matrix across folds
    avg_cm = np.mean(all_fold_cms, axis=0)
    
    # Plot average confusion matrix
    avg_title = f"Average Confusion Matrix - Mode={mode} | Across {len(all_fold_cms)} Folds"
    avg_save_path = os.path.join(confusion_dir, "average_confusion_matrix.png")
    plot_confusion_matrix(avg_cm, class_labels, avg_title, avg_save_path)
    
    print(f"✓ Average confusion matrix saved\n")
    
    # Compute aggregate statistics
    all_preds_combined = np.concatenate(all_preds_per_fold)
    all_labels_combined = np.concatenate(all_labels_per_fold)
    
    # Save detailed results
    results = {
        'mode': mode,
        'num_folds': len(all_fold_cms),
        'num_classes': num_classes,
        'test_samples': len(test_df),
        'class_labels': class_labels,
        'average_cm': avg_cm.tolist(),
        'overall_accuracy': np.mean(all_preds_combined == all_labels_combined),
    }
    
    results_path = os.path.join(confusion_dir, "confusion_matrix_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {results_path}")
    print(f"\n  Mode: {mode}")
    print(f"  Num Folds: {len(all_fold_cms)}")
    print(f"  Overall Test Accuracy: {results['overall_accuracy']:.4f}")
    print(f"  Test Samples: {len(test_df)}")
    
    # Summary statistics
    print(f"\n  Per-class accuracy (from average CM):")
    for i, label in enumerate(class_labels):
        if i < len(avg_cm):
            class_acc = avg_cm[i, i]  # 대각선 원소 = 해당 클래스 정확도
            print(f"    {label:3s}: {class_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrices for test set")
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
    
    for mode in modes:
        generate_confusion_matrices_for_mode(mode, args.output_dir, k_folds=5)
    
    print(f"\n{'='*80}")
    print("Confusion matrix generation completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
