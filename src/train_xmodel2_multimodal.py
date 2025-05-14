"""
Train Xmodel-2 with BLIP-2 + Q-Former Vision Encoder (Multimodal: Text + Image)

Requirements:
- transformers >= 4.36.0
- torch
- timm
- pillow
- pandas

Dataset CSV must have columns: image_path, source, target
"""
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2Model, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AdamW

# === Config ===
BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"  # or checkpoint compatible with Xmodel-2
XMODEL2_MODEL = "XiaoduoAILab/XmodelLM1.5"  # path or HF repo
CSV_PATH = "data/train.csv"  # must have image_path, source, target
OUTPUT_DIR = "models/Xmodel2_multimodal"
BATCH_SIZE = 4
EPOCHS = 3
LR = 3e-5
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Dataset ===
class MultimodalDataset(Dataset):
    def __init__(self, csv_path, processor, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        text = str(row['source'])
        target = str(row['target'])
        # BLIP-2 processor returns pixel_values for vision encoder
        image_inputs = self.processor(images=image, return_tensors="pt")
        # Tokenize text prompt and target
        text_inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        target_inputs = self.tokenizer(target, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        return {
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'labels': target_inputs['input_ids'].squeeze(0)
        }

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# === Main Training Script ===
def main():
    print(f"Loading BLIP-2 + Q-Former from {BLIP2_MODEL}")
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL)
    blip2 = Blip2Model.from_pretrained(BLIP2_MODEL).to(DEVICE)
    print(f"Loading Xmodel-2 from {XMODEL2_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(XMODEL2_MODEL, trust_remote_code=True)
    xmodel2 = AutoModelForCausalLM.from_pretrained(XMODEL2_MODEL, trust_remote_code=True).to(DEVICE)

    dataset = MultimodalDataset(CSV_PATH, processor, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(xmodel2.parameters(), lr=LR)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    xmodel2.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            pixel_values = batch['pixel_values'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            # 1. Encode image to get image embeddings (Q-Former)
            with torch.no_grad():
                vision_outputs = blip2.vision_model(pixel_values=pixel_values)
                image_embeds = vision_outputs.last_hidden_state
                qformer_outputs = blip2.qformer(query_embeds=image_embeds)
                image_tokens = qformer_outputs.last_hidden_state
            # 2. Concatenate image tokens with text input (image prefix)
            # (This step may need to be adapted for your Xmodel-2 input format)
            # Here, we simply prepend image tokens to input_ids
            # You may need to use special tokens or follow Xmodel-2's multimodal input convention
            # For demonstration, we flatten image_tokens and concatenate to input_ids
            # (In practice, check your model's multimodal API)
            multimodal_input = torch.cat([image_tokens.flatten(1), input_ids], dim=1)
            # 3. Forward pass
            outputs = xmodel2(input_ids=multimodal_input, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xmodel2.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
