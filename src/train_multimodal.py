# train_multimodal.py
"""
Training script for Vision+LLM (ZombitxLM + CLIP) on image-text dataset
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from src.vision_model import VisionLanguageModel
from src.multimodal_dataloader import ImageTextDataset

# Config
LLM_PATH = "models/Zombit/ZombitxLM/instruct"  # Path to ZombitxLM production model
VISION_ENCODER = "openai/clip-vit-large-patch14"
PARQUET_PATH = "DatasetDownload/lintw_VL-Health/Instruct_Fine_Tuning/Comprehension/train-00000-of-00009.parquet"  # ตัวอย่าง path จริง
IMAGE_ROOT = "DatasetDownload/lintw_VL-Health/Instruct_Fine_Tuning/Comprehension/images"   # ปรับตาม dataset จริง
BATCH_SIZE = 2
EPOCHS = 1
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset & Dataloader
train_dataset = ImageTextDataset(PARQUET_PATH, IMAGE_ROOT, text_col="text", image_col="image")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = VisionLanguageModel(LLM_PATH, vision_encoder_name=VISION_ENCODER).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

# Training loop (captioning-style loss)
model.train()
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, texts in pbar:
        images = list(images)
        texts = list(texts)
        images = [img.to(DEVICE) if hasattr(img, 'to') else img for img in images]
        outputs = model(images, texts)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})
print("Training finished.")
