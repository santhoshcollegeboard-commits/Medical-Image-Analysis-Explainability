"""
Training pipeline for Deep Learning Medical Image Analysis.
Includes class-weighted Cross-Entropy loss to combat clinical class imbalance,
validation monitoring, model checkpointing, and training curve visualization.
"""

import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from .dataset import get_medical_dataloaders
    from .model import MedicalResNet
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from dataset import get_medical_dataloaders
    from model import MedicalResNet


def plot_medical_training_curves(history: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curve
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#2563eb', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#dc2626', linewidth=2, linestyle='--')
    ax1.set_title('Medical Weighted Cross-Entropy Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True, facecolor='white')

    # Accuracy Curve
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], label='Train Accuracy', color='#059669', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], label='Val Accuracy', color='#d97706', linewidth=2, linestyle='--')
    ax2.set_title('Screening Accuracy (%) vs Epochs', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True, facecolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Medical training curves saved to: {save_path}")


def train_medical_model(
    data_flag: str = 'pneumoniamnist',
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    output_dir: str = './development',
    fig_dir: str = './github/examples'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Medical Training on device: {device}")

    train_loader, val_loader, test_loader, label_dict, class_weights = get_medical_dataloaders(
        data_flag=data_flag, batch_size=batch_size
    )

    print(f"Computed Class Weights: Normal={class_weights[0]:.2f}, Pneumonia={class_weights[1]:.2f}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = MedicalResNet(in_channels=1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    checkpoint_dir = os.path.join(output_dir, 'model_checkpoints')
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Starting Training for {epochs} Epochs on {data_flag}...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Val
        model.eval()
        val_loss_run, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss_run += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_total += targets.size(0)
                val_correct += preds.eq(targets).sum().item()

        val_loss = val_loss_run / val_total
        val_acc = val_correct / val_total

        scheduler.step(val_acc)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(checkpoint_dir, "medical_resnet_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_weights': class_weights,
                'label_dict': label_dict
            }, best_path)

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%")

    with open(os.path.join(logs_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=4)

    plot_path = os.path.join(fig_dir, "training_curves.png")
    plot_medical_training_curves(history, plot_path)

    return history, best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train_medical_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
