import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models, transforms


def get_class_balanced_weights(dataset):
    all_labels = [batch['labels'] for batch in dataset]
    all_labels = torch.cat(all_labels, dim=0)

    pos_counts = all_labels.sum(dim=0)
    neg_counts = (1 - all_labels).sum(dim=0)

    class_weights = neg_counts / (pos_counts + 1e-6)
    return class_weights.to(torch.float)



class DenseNet121(nn.Module):

    def __init__(self):
        super(DenseNet121, self).__init__()

        NUM_CLASSES = 15

        self.densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        kernel_count = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(
            nn.Linear(kernel_count, NUM_CLASSES),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
