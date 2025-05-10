from glob import glob
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch


class ChestXrayDataset(Dataset):
    def __init__(self, split_file, image_root_dir, bbox_csv, metadata_csv, transform=None, target_size=(224, 224)):

        # get all image names from train/val list or test list
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f]

        self.image_root_dir = image_root_dir

        self.filepath_dict = self._build_index()

        # ✅ Filter only image_names that are actually available on disk
        valid_image_names = [name for name in self.image_names if name in self.filepath_dict]
        if len(valid_image_names) < len(self.image_names):
            missing = set(self.image_names) - set(valid_image_names)
            print(f"⚠️ Skipping {len(missing)} missing images from {split_file}")
        self.image_names = valid_image_names

        # default transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

        # load CSVs
        self.bbox_df = pd.read_csv(bbox_csv)
        self.meta_df = pd.read_csv(metadata_csv)

        # include only images in the split
        self.meta_df = self.meta_df[self.meta_df['Image Index'].isin(self.image_names)].copy()

        # identify label columns
        self.label_columns = sorted([
            col for col in self.meta_df.columns
            if col not in ['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID',
                           'Patient Age', 'Patient Gender', 'View Position',
                           'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]',
                           'Unnamed: 11']
        ])

        # build {filename: [disease, [x, y, w, h]]}
        self.bbox_dict = {
            row['Image Index']: [row['Finding Label'], [row['Bbox [x'], row['y'], row['w'], row['h]']]]
            for _, row in self.bbox_df.iterrows()
        }

        # build {filename: diseases}
        self.label_dict = {
            row['Image Index']: row[self.label_columns].values.astype('float32')
            for _, row in self.meta_df.iterrows()
        }


    def _build_index(self):
        index = {}
        for path in glob(os.path.join(self.image_root_dir, '**/*.png'), recursive=True):
            filename = os.path.basename(path)
            index[filename] = path
        return index
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        filename = self.image_names[idx]
        image_path = self.filepath_dict[filename]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Default values if missing
        labels = self.label_dict.get(filename)
        if labels is None:
            labels = torch.zeros(len(self.label_columns), dtype=torch.float32)
        labels = torch.tensor(labels)

        bbox = self.bbox_dict.get(filename, None)
        if bbox is None:
            bbox = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        return {
            'image': image,
            'labels': labels,
            'bbox': bbox,
            'image_name': filename
        }
    