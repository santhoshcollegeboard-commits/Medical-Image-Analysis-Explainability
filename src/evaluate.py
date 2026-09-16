"""
Clinical evaluation and explainability pipeline for Medical Image Analysis.
Computes Sensitivity (Recall), Specificity, Precision, F1-Score, ROC-AUC,
Confusion Matrix, and generates Grad-CAM heatmaps over thoracic radiographs.
"""

import os
import json
import argparse
from typing import Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_fscore_support
)

try:
    from .dataset import get_medical_dataloaders
    from .model import MedicalResNet
    from .gradcam import GradCAM
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from dataset import get_medical_dataloaders
    from model import MedicalResNet
    from gradcam import GradCAM


def plot_medical_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, square=True
    )
    plt.title('Medical Screening Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Predicted Pathology', fontsize=11, labelpad=8)
    plt.ylabel('Ground Truth Pathology', fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(targets: np.ndarray, probs: np.ndarray, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(targets, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'MedicalResNet (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#9ca3af', lw=1.5, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', pad=10)
    plt.legend(loc="lower right", frameon=True, facecolor='white')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {save_path}")
    return roc_auc


def generate_gradcam_visualizations(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    label_dict: dict,
    device: torch.device,
    save_path: str,
    num_cases: int = 4
):
    """
    Generates a side-by-side comparison of original radiographs, Grad-CAM heatmaps,
    and overlaid anatomical visual attributions.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gradcam = GradCAM(model)

    test_inputs, test_targets = next(iter(loader))
    test_targets = test_targets.squeeze().numpy()

    fig, axes = plt.subplots(num_cases, 3, figsize=(11, 3.2 * num_cases))

    for idx in range(num_cases):
        img_tensor = test_inputs[idx:idx+1].to(device)
        true_lbl = test_targets[idx]

        heatmap, pred_cls, conf = gradcam.generate_heatmap(img_tensor)

        raw_img = test_inputs[idx, 0].numpy()
        # Denormalize from mean=0.5, std=0.5 to [0, 1]
        raw_display = np.clip(raw_img * 0.5 + 0.5, 0, 1)

        overlay = gradcam.overlay_heatmap(heatmap, raw_display, alpha=0.5)

        # 1. Original Radiograph
        axes[idx, 0].imshow(raw_display, cmap='gray')
        axes[idx, 0].set_title(f"Case {idx+1}: Ground Truth [{label_dict[true_lbl]}]", fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')

        # 2. Raw Grad-CAM Heatmap
        axes[idx, 1].imshow(heatmap, cmap='jet')
        axes[idx, 1].set_title(f"Grad-CAM Saliency Map", fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')

        # 3. Overlaid Attribution
        axes[idx, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        match_color = '#16a34a' if pred_cls == true_lbl else '#dc2626'
        axes[idx, 2].set_title(
            f"Prediction: {label_dict[pred_cls]} ({conf*100:.1f}%)",
            fontsize=10, fontweight='bold', color=match_color
        )
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM explanations saved to: {save_path}")


def evaluate_medical_model(
    checkpoint_path: str,
    data_flag: str = 'pneumoniamnist',
    batch_size: int = 64,
    fig_dir: str = './github/examples',
    log_dir: str = './development/logs'
) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Medical Evaluation on device: {device}")

    _, _, test_loader, label_dict, _ = get_medical_dataloaders(data_flag=data_flag, batch_size=batch_size)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MedicalResNet(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.squeeze().long()
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Clinical screening metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    os.makedirs(fig_dir, exist_ok=True)
    roc_path = os.path.join(fig_dir, "roc_curve.png")
    roc_auc = plot_roc_curve(all_targets, all_probs, roc_path)

    cm_path = os.path.join(fig_dir, "confusion_matrix.png")
    plot_medical_confusion_matrix(cm, [label_dict[0], label_dict[1]], cm_path)

    gradcam_path = os.path.join(fig_dir, "gradcam_explanations.png")
    generate_gradcam_visualizations(model, test_loader, label_dict, device, gradcam_path)

    metrics = {
        'benchmark_dataset': data_flag,
        'total_test_samples': int(len(all_targets)),
        'accuracy': float(round(accuracy, 4)),
        'sensitivity_recall': float(round(sensitivity, 4)),
        'specificity': float(round(specificity, 4)),
        'precision': float(round(precision, 4)),
        'f1_score': float(round(f1, 4)),
        'roc_auc': float(round(roc_auc, 4)),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    }

    print("\n" + "="*55)
    print("CLINICAL EVALUATION RESULTS (PneumoniaMNIST TEST SET)")
    print("="*55)
    print(f"Overall Accuracy:       {metrics['accuracy']*100:.2f}%")
    print(f"Sensitivity (Recall):   {metrics['sensitivity_recall']*100:.2f}% (Crucial for screening)")
    print(f"Specificity:            {metrics['specificity']*100:.2f}%")
    print(f"Precision:              {metrics['precision']:.4f}")
    print(f"F1-Score:               {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:                {metrics['roc_auc']:.4f}")
    print(f"True Positives:         {tp} / {tp+fn}")
    print(f"True Negatives:         {tn} / {tn+fp}")
    print("="*55)

    os.makedirs(log_dir, exist_ok=True)
    out_json = os.path.join(log_dir, "medical_metrics.json")
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Medical metrics saved to: {out_json}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate medical image analysis model")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to trained checkpoint (.pth). If omitted, searches default local checkpoint paths.")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        default_candidates = [
            '../../development/model_checkpoints/medical_resnet_best.pth',
            '../development/model_checkpoints/medical_resnet_best.pth',
            './development/model_checkpoints/medical_resnet_best.pth',
            'model_checkpoints/medical_resnet_best.pth'
        ]
        for c in default_candidates:
            if os.path.exists(c):
                ckpt = c
                break

    if ckpt is None or not os.path.exists(ckpt):
        print("Note: No local checkpoint specified. Provide --checkpoint <path/to/checkpoint.pth>")
        print("Example: python src/evaluate.py --checkpoint ../development/model_checkpoints/medical_resnet_best.pth")
    else:
        evaluate_medical_model(checkpoint_path=ckpt)
