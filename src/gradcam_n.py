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
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device).eval()

        # extract features (remove classifier)
        self.feature_extractor = self.model.densenet121.features
        self.classifier = self.model.densenet121.classifier

        # define activations and gradients
        self.activations = None
        self.gradients = None

        # register hooks on the last conv block inside feature_extractor
        last_conv_layer = self.feature_extractor[-1]  # or whichever submodule is final
        last_conv_layer.register_forward_hook(self._save_activation)
        last_conv_layer.register_full_backward_hook(self._save_gradient)

        # define transforms
        self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
        ])

    def _save_activation(self, module, input, output):
        # Called during forward pass; output is shape [B, C, Hf, Wf]
        self.activations = output.detach()  # keep a copy on CPU/GPU

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] is the gradients of that layer’s output w.r.t. loss
        self.gradients = grad_output[0].detach()


    def generate(self, img_path, output_path, transCrop=224, target_class=None):
        # load image and preprocess the image
        pil_img = Image.open(img_path).convert('RGB')
        inp = self.transform(pil_img)
        inp = inp.unsqueeze(0).to(self.device) # [1, 3, 224, 224]

        # forward pass through CNN
        self.feature_extractor.to(self.device).eval()
        self.classifier.to(self.device).eval()

        # zero previous gradients
        self.feature_extractor.zero_grad()
        self.classifier.zero_grad()

        # extract feature maps via forward hook
        feats = self.feature_extractor(inp) # [1, C, Hf, Wf]

        # push through classifier to get logits
        feats_flat = feats.view(feats.size(0), -1)
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, (1,1)).view(feats.size(0), -1)
        logits = self.classifier(pooled) # [1, num_classes]

        # choose a target class index
        if target_class is None:
            # pick whichever logit is highest
            target_class = torch.argmax(logits, dim=1).item()

        # backward pass on that single scalar
        self.feature_extractor.zero_grad()
        self.classifier.zero_grad()
        scalar_logit = logits[0, target_class]
        scalar_logit.backward(retain_graph=True)

        # set activation and gradients
        activs = self.activations[0]  # [C, Hf, Wf]
        grads  = self.gradients[0]    # [C, Hf, Wf]

        # global‐avg‐pool the gradients to get a weight per channel
        channel_weights = torch.mean(grads.view(grads.size(0), -1), dim=1)  # shape [C]

        # compute raw heatmap
        heatmap = torch.zeros_like(activs[0])
        for i in range(activs.size(0)): # iterate over C channels
            heatmap += channel_weights[i] * activs[i]
        heatmap = torch.relu(heatmap) # only keep positives

        
        # move to cpu and convert to numpy
        np_hm = heatmap.cpu().detach().numpy()
        if np.max(np_hm) > 0:
            np_hm = np_hm / np.max(np_hm)

        # upsample to (transcrop x transcrop)
        np_hm_resized = cv2.resize(np_hm, (transCrop, transCrop))

        # read in and resize original image
        original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        original_img = cv2.resize(original_img, (transCrop, transCrop))


        # colorize the heatmap with JET
        heatmap_uint8 = np.uint8(255 * np_hm_resized)  # [0,255], dtype=uint8
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR colormap

        # blend heatmap with original
        blended = cv2.addWeighted(heatmap_color, 0.4, original_img, 0.6, 0)

        # save image
        cv2.imwrite(output_path, blended)


ckpt_path ='models/densenet121_best.pt'
img_path = 'data/xray_images/images_005/images/00009285_000.png'
output_path = 'gradcam_images/trial.png'

h = GradCAM(ckpt_path=ckpt_path)
h.generate(img_path=img_path, output_path=output_path)
