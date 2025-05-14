# multimodal_dataloader.py
"""
Dataloader for image-text pair dataset (e.g. lintw/VL-Health)
"""
import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ImageTextDataset(Dataset):
    def __init__(self, file_path, image_root, text_col="text", image_col="image"):
        # Support both CSV and Parquet
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            self.data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        self.image_root = image_root
        self.text_col = text_col
        self.image_col = image_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, row[self.image_col])
        image = Image.open(image_path).convert("RGB")
        text = row[self.text_col]
        return image, text
