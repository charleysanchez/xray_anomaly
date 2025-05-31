import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # save activations of target conv layer
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # save activation gradients
            self.gradients = grad_out[0].detach()

        # register hooks
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        input_tensor: 1xCxHxW normalized tensor on same device as model
        class_idx: index of target class; if None, uses top predicted
        returns: HxW heatmap normalized [0,1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            # take predicted class
            class_idx = output.argmax(dim=1).item()
        # create one hot and backprop
        one_hot = torch.zeros_like(output, device=output.device)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # weights: global-average-pool of gradients
        grads = self.gradients[0] # CxWxH
        weights = grads.mean(dim=(1,2), keepdim=True) # Cx1x1

        # weighted combination of activations
        activations = self.activations[0] # CxWxH
        cam = (weights * activations).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)

        # normalize
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()