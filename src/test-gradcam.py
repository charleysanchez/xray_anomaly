from torchvision import models, transforms
import torch
from PIL import Image
import cv2
from gradcam import GradCAM
import numpy as np
import os

NUM_CLASSES = 14
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load model ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
checkpoint = torch.load('models/resnet50_epoch_10.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

# --- Set up GradCAM ---
target_layer = model.layer4[-1]
grad_cam = GradCAM(model, target_layer)

# --- Preprocess your image ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
img_path = 'data/xray_images/images_001/images/00000001_000.png'
img_name = os.path.basename(img_path)
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0).to(device)

# --- Generate and save overlay ---
cam = grad_cam.generate(input_tensor)
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
orig = np.array(img.resize((224, 224)))
overlay = np.uint8(0.5 * heatmap + 0.5 * orig)
out_path = f'gradcam_images/{img_name}'
cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"Saved Grad-CAM overlay to {out_path}")

grad_cam.remove_hooks()
