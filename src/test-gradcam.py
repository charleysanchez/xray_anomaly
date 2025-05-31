from torchvision import models, transforms
import torch
from PIL import Image
import cv2
from gradcam import GradCAM
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

NUM_CLASSES = 15
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# --- Load model ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

checkpoint = torch.load(
    'models/resnet50_epoch_61.pt',
    map_location=device,
    weights_only=False
)

# fix up any fc.1 → fc name mismatches
raw_sd = checkpoint['model_state_dict']
fixed_sd = {}
for k, v in raw_sd.items():
    if k.startswith('fc.1.'):
        new_key = k.replace('fc.1.', 'fc.')
    else:
        new_key = k
    fixed_sd[new_key] = v

model.load_state_dict(fixed_sd)
model.to(device).eval()

# --- Set up GradCAM ---
target_layer = model.layer4[-1].conv3
grad_cam = GradCAM(model, target_layer)

# --- Preprocess your image ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
img_path = 'data/xray_images/images_004/images/00006657_000.png'
img_name = os.path.basename(img_path)
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0).to(device)

# ensure output folder
os.makedirs('gradcam_images', exist_ok=True)

# --- Disease names mapping ---
disease_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

# --- Forward pass to get predicted class ---
with torch.no_grad():
    output = model(input_tensor)
pred_idx = output.argmax(dim=1).item()

disease_name = disease_names[pred_idx]
print(f"Predicted disease: {disease_name} (class {pred_idx})")

df = pd.read_csv('data/xray_images/Data_Entry_2017.csv')
actual_diseases = df.loc[df['Image Index'] == img_name, 'Finding Labels']
print(f"Actual diseases: {actual_diseases}")

# --- Generate and visualize single-class Grad-CAM ---
cam = grad_cam.generate(input_tensor, class_idx=pred_idx)

# Overlay parameters
orig = np.array(img.resize((224, 224)))

# Resize cam to image dimensions
heatmap = cv2.resize(cam, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(orig)
im = ax.imshow(heatmap, cmap='jet', alpha=0.5)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Grad-CAM Intensity')
ax.set_title(f"Grad-CAM for {disease_name}")
ax.axis('off')

# Save
save_path = f"gradcam_images/{os.path.splitext(img_name)[0]}_{disease_name}.png"
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved Grad-CAM overlay with colorbar to {save_path}")

# --- Cleanup hooks ---
grad_cam.remove_hooks()