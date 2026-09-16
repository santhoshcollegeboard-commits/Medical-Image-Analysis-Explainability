"""
Single-image clinical inference and visual explainability script.
Loads a thoracic radiograph, executes MedicalResNet forward pass,
computes pathological probabilities, and generates a Grad-CAM attribution heatmap.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

try:
    from .model import MedicalResNet
    from .gradcam import GradCAM
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from model import MedicalResNet
    from gradcam import GradCAM

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


def predict_radiograph(
    image_path: str,
    checkpoint_path: str = None,
    output_vis_path: str = 'examples/prediction_gradcam.png',
    device: str = None
):
    dev = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input radiograph not found at: {image_path}")

    # Load image as grayscale [0, 255]
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        raw_img = np.array(Image.open(image_path).convert('L'))

    # Standardize to 28x28 (PneumoniaMNIST resolution)
    resized_img = cv2.resize(raw_img, (28, 28))
    norm_img = resized_img.astype(np.float32) / 255.0
    tensor = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)

    # Initialize model
    model = MedicalResNet(in_channels=1, num_classes=2).to(dev)

    # Checkpoint fallback
    if checkpoint_path is None:
        default_candidates = [
            '../../development/model_checkpoints/medical_resnet_best.pth',
            '../development/model_checkpoints/medical_resnet_best.pth',
            './development/model_checkpoints/medical_resnet_best.pth',
            'model_checkpoints/medical_resnet_best.pth'
        ]
        for c in default_candidates:
            if os.path.exists(c):
                checkpoint_path = c
                break

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded clinical weights from: {checkpoint_path}")
    else:
        print("Running with initialized model weights (no checkpoint specified).")

    model.eval()

    # Forward pass
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]

    print("=" * 50)
    print("CLINICAL RADIOGRAPH SCREENING REPORT")
    print("=" * 50)
    print(f"Input Radiograph:    {image_path}")
    print(f"Pathology Finding:   {pred_label}")
    print(f"Confidence:          {confidence * 100:.2f}%")
    print(f"Probability Normal:    {probs[0] * 100:.2f}%")
    print(f"Probability Pneumonia: {probs[1] * 100:.2f}%")
    print("=" * 50)

    # Generate Grad-CAM attribution heatmap
    gradcam = GradCAM(model)
    heatmap, _, _ = gradcam.generate_heatmap(tensor, target_class=pred_idx)

    # Overlay heatmap on original image
    disp_img = cv2.resize(raw_img, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    disp_bgr = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(disp_bgr, 0.65, heatmap_colored, 0.35, 0)

    color = (0, 0, 255) if pred_label == 'PNEUMONIA' else (0, 255, 0)
    cv2.putText(overlay, f"{pred_label} ({confidence * 100:.1f}%)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    combined = np.hstack([disp_bgr, overlay])
    os.makedirs(os.path.dirname(output_vis_path), exist_ok=True)
    cv2.imwrite(output_vis_path, combined)
    print(f"Grad-CAM visual explanation saved to: {output_vis_path}")

    return {
        "finding": pred_label,
        "confidence": float(confidence),
        "normal_prob": float(probs[0]),
        "pneumonia_prob": float(probs[1]),
        "visualization_path": output_vis_path
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single-radiograph screening and Grad-CAM explainability")
    parser.add_argument('--image', type=str, required=True, help="Path to input chest radiograph")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to model weights (.pth)")
    parser.add_argument('--output', type=str, default='examples/prediction_gradcam.png',
                        help="Path to save Grad-CAM visualization")
    args = parser.parse_args()

    predict_radiograph(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_vis_path=args.output
    )
