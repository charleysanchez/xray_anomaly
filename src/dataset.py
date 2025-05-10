from glob import glob
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ChestXrayDataset(Dataset):
    def __init__(self, split_file, image_root_dir, bbox_csv, metadata_csv, transform=None, target_size=(224, 224)):

        # get all image names from train/val list or test list
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f]
        self.image_root_dir = image_root_dir

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
                           'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]']
        ])

        # build {filename: path}
        self.filepath_dict = self._build_index()

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
        for subfolder in os.listdir(self.image_root_dir):
            full_subfolder = os.path.join(self.image_root_dir, subfolder)
            if os.path.isdir(full_subfolder):
                for path in glob(os.path.join(full_subfolder, '*.png')):
                    filename = os.path.basename(path)
                    index[filename] = path
        return index
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        filename = self.image_names[idx]
        image_path = self.filepath_dict[filename]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(Image)

        bbox = self.bbox_dict.get(filename, None) # list of [Disease, [x, y, w, h]]
        labels = self.label_dict.get(filename)

        return {
            'image': image,
            'labels': labels,
            'bbox': bbox,
            'image_name': filename
        }
    