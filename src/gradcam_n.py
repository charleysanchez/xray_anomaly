import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from model import DenseNet121

#----------------------------------------------------------------------------------
#------ Class to generate gradcam ----------

class GradCAM():
    """
    Initialize object
    model_path: path to the trained model
    architecture: pretrained model architecture (default densenet121)
    num_classes: number of potential output classes (default 15)
    """

    def __init__(self, ckpt_path, architecture='densenet121', num_classes=15):

        # define model
        if architecture == 'densenet121':
            self.model = DenseNet121()

        # set default device and send model to device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # load checkpoint
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # extract features (remove classifier)
        self.model = self.model.densenet121.features
        self.model.eval()

        # initialize the weights
        self.weights = list(self.model.parameters())[-2]

        # define transforms
        self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
        ])


    def generate(self, img_path, output_path):
        # load image and preprocess the image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze_(0).to(self.device) # [1, 3, 224, 224]

        # forward pass through CNN
        self.model.to(self.device).eval()
        with torch.no_grad():
            feature_maps = self.model(image) # [1, C, Hf, Wf]

        # compute raw heatmap
        heatmap = None
        for i in range(feature_maps.shape[1]): # iterate over C channels
            fmap = feature_maps[0, i, :, :]    # [Hf, Wf]
            w = self.weights[i]                # scalar weight for channel i
            if heatmap is None:
                heatmap = w * fmap
            else:
                heatmap += w * fmap

        
        # move to cpu and convert to numpy
        np_heatmap = heatmap.cpu().numpy()

        # read in and resize original image
        original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        original_img = cv2.resize(original_img, (256, 256))

        # normalize and resize heatmap
        cam = np_heatmap / np.max(np_heatmap)
        cam = cv2.resize(cam, (256, 256))

        # apply JET colormap to normalized heatmap
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        # blend heatmap with original
        blended = cv2.addWeighted(heatmap_color, 0.5, original_img, 0.5, 0)

        # save image
        cv2.imwrite(output_path, blended)
