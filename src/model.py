import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

def get_class_balanced_weights(dataset):
    targets = dataset['labels']
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)

    return torch.tensor(class_weights, dtype=torch.float)