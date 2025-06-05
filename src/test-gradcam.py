import json
from torchvision import models, transforms
import torch
from PIL import Image
import cv2, numpy as np
from gradcam import GradCAM
import matplotlib.pyplot as plt
import os
import pandas as pd

def test_gradcam(image_paths):
    NUM_CLASSES = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    checkpoint = torch.load('models/resnet50_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # Hook onto the last conv layer:
    target_layer = model.layer4[-1].conv3
    grad_cam = GradCAM(model, target_layer)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    disease_names = [
        'Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
        'Emphysema','Fibrosis','Hernia','Infiltration','Mass',
        'No Finding','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax'
    ]

    # -----------------------------
    # LOAD PER-CLASS THRESHOLDS
    # -----------------------------
    with open('per_class_thresholds.json', 'r') as f:
        thresholds_dict = json.load(f)
    # Convert to a NumPy array in the same order as disease_names:
    thresholds = np.array([thresholds_dict[name] for name in disease_names], dtype=float)

    os.makedirs('gradcam_images', exist_ok=True)

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_name = os.path.basename(img_path)
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)           # raw logits [1,15]
            probs = torch.sigmoid(logits)
            probs = probs.cpu().numpy()[0]               # final NumPy array of length 15

        # print(probs)
        # pick all classes where prob ≥ its own threshold
        positive_classes = [i for i,p in enumerate(probs) if p >= thresholds[i]]
        if len(positive_classes) == 0:
            # Fallback: if no class exceeds threshold, you could pick the highest-probability class:
            positive_classes = [int(probs.argmax())]

        for class_idx in positive_classes:
            cam = grad_cam.generate(input_tensor, class_idx=class_idx)

            if cam.max() < 1e-4:
                print(f"Warning: almost zero GradCAM for class {class_idx} on {img_name}")

            cam_uint8 = np.uint8(255 * cam)
            heatmap = cv2.resize(cam_uint8, (224,224), interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap.astype(np.float32) / 255.0

            orig = np.array(img.resize((224,224)))
            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(orig)
            im = ax.imshow(heatmap, cmap='jet', alpha=0.5)
            ax.axis('off')
            ax.set_title(f"{disease_names[class_idx]} (p={probs[class_idx]:.2f}, thr={thresholds[class_idx]:.2f})")

            save_name = f"{os.path.splitext(img_name)[0]}_{disease_names[class_idx]}.png"
            plt.savefig(f"gradcam_images/{save_name}", bbox_inches='tight', dpi=150)
            plt.close()
            grad_cam.remove_hooks()

        df = pd.read_csv("data/xray_images/Data_Entry_2017.csv")
        actual_classes = df.loc[df["Image Index"] == img_name, 'Finding Labels'].item().split('|')
        class_probs = {c: probs[disease_names.index(c)] for c in actual_classes}
        print(
            f"Saved GradCAM for: {img_name} → "
            f"predicted {[f'{disease_names[p]}: {probs[p]}' for p in positive_classes]}; "
            f"actual {class_probs}"
        )



if __name__ == '__main__':
    image_paths = [
        'data/xray_images/images_004/images/00006657_000.png',
        'data/xray_images/images_004/images/00006658_000.png',
        'data/xray_images/images_004/images/00006663_000.png',
        'data/xray_images/images_004/images/00006663_001.png',
        'data/xray_images/images_004/images/00006663_002.png',
        'data/xray_images/images_004/images/00006663_003.png',
        'data/xray_images/images_004/images/00006663_004.png',
        'data/xray_images/images_004/images/00006731_000.png',
        'data/xray_images/images_004/images/00006731_001.png',
        'data/xray_images/images_004/images/00006731_002.png',
        'data/xray_images/images_004/images/00006731_003.png',
        'data/xray_images/images_004/images/00006731_004.png',
        'data/xray_images/images_004/images/00006731_005.png',
        'data/xray_images/images_004/images/00006679_010.png',
        'data/xray_images/images_004/images/00006736_001.png',
        'data/xray_images/images_004/images/00006717_007.png',
        'data/xray_images/images_004/images/00006714_001.png',
        'data/xray_images/images_004/images/00006713_014.png',
        'data/xray_images/images_004/images/00006713_013.png',

    ]

    test_gradcam(image_paths)