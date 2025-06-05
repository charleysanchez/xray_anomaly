import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()  # store activations

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; the gradients w.r.t. the forward output
            self.gradients = grad_out[0].detach()

        # attach hooks to the target layer
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        1) Run a forward pass to get raw logits
        2) Convert to probabilities (sigmoid) and detach for NumPy
        3) If no class_idx passed, pick top‐1 logit
        4) Backprop from that single logit to get gradients
        5) Compute Grad-CAM heatmap from activations & gradients
        6) Return a [H×W] float heatmap (values in [0,1])
        """
        # ---- 1) Forward pass (still traced for hooks) ----
        logits = self.model(input_tensor)  # [1, NUM_CLASSES], requires_grad

        # ---- 2) Sigmoid → detach → numpy for thresholding ----
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]  # [NUM_CLASSES]

        # ---- 3) Determine which class to backpropagate from ----
        if class_idx is None:
            class_idx = int(np.argmax(probs))

        # Construct one‐hot for the chosen class
        one_hot = torch.zeros_like(logits, device=logits.device)
        one_hot[0, class_idx] = 1.0

        # ---- 4) Backpropagate from that one logit ----
        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)

        # ---- 5) Build the Grad-CAM heatmap ----
        activations = self.activations[0]  # [C, H, W]
        gradients   = self.gradients[0]    # [C, H, W]
        weights     = gradients.mean(dim=(1, 2), keepdim=True)  # [C,1,1]
        cam = (weights * activations).sum(dim=0)                # [H, W]
        cam = torch.relu(cam)                                   # remove negative values
        cam = cam.cpu().numpy()

        # Normalize to [0,1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam  # numpy array shape [H, W]
