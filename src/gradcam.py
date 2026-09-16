"""
Gradient-Weighted Class Activation Mapping (Grad-CAM) implementation.
Produces visual explanations for decisions made by the medical convolutional network.
Highlights specific anatomical regions (pulmonary infiltrates / consolidations) influencing the prediction.
"""

from typing import Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM engine for visual attribution and anatomical explainability.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates a normalized 2D Grad-CAM heatmap for the specified target class.
        If target_class is None, uses the class with the highest predicted probability.
        """
        self.model.zero_grad()

        # Forward pass with gradient tracking on features
        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass on target score
        score = output[0, target_class]
        score.backward()

        # Retrieve pooled gradients and layer activations
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()

        if gradients is None or activations is None:
            raise RuntimeError("Failed to extract activations or gradients for Grad-CAM.")

        # Global average pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Weight activations by pooled gradients
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Channel-wise summation and ReLU to retain only positive influences
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        # Normalize heatmap to [0, 1]
        heatmap = heatmap.detach().cpu().numpy()
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap, target_class, confidence

    @staticmethod
    def overlay_heatmap(
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlays the 2D heatmap onto the original grayscale or RGB radiograph.
        """
        h, w = original_image.shape[:2]
        resized_heatmap = cv2.resize(heatmap, (w, h))

        heatmap_uint8 = np.uint8(255 * resized_heatmap)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        if len(original_image.shape) == 2 or original_image.shape[2] == 1:
            base_rgb = cv2.cvtColor(np.uint8(255 * original_image), cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = np.uint8(255 * original_image)

        overlayed = cv2.addWeighted(color_heatmap, alpha, base_rgb, 1 - alpha, 0)
        return overlayed
