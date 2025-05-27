import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

class ClassBalancedCELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_bce = bce * self.class_weights.to(inputs.device)
        return weighted_bce.mean()



def get_class_balanced_weights(dataset, beta=0.9990):
    targets = dataset['labels']
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)

    return torch.tensor(class_weights, dtype=torch.float)