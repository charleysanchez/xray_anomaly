import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

def get_class_balanced_weights(dataset):
    all_labels = [batch['labels'] for batch in dataset]
    all_labels = torch.cat(all_labels, dim=0)

    pos_counts = all_labels.sum(dim=0)
    neg_counts = (1 - all_labels).sum(dim=0)

    class_weights = neg_counts / (pos_counts + 1e-6)
    return class_weights.to(torch.float)