import numpy as np
import torch
import torch.nn as nn

class ClassBalancedBCELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_bce = bce * self.class_weights.to(inputs.device)
        return weighted_bce.mean()

def compute_class_counts(dataset):
    all_labels = [sample['labels'].numpy() for sample in dataset]
    all_labels = np.stack(all_labels)
    return np.sum(all_labels, axis=0)


def get_class_balanced_weights(class_counts, beta=0.9990):
    effective_num = 1.0 * np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / np.sum(weights) * len(class_counts)
    return torch.tensor(weights, dtype=torch.float32)