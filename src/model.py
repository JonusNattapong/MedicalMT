from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from safetensors.torch import save_file as save_safetensors
from typing import Optional
import os

class MedMTModel:
    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def translate(self, src_texts, max_length=256):
        inputs = self.tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def save(self, save_dir: str, use_safetensors: bool = False):
        # Always save in 'models' directory
        save_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(save_dir, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)
        if use_safetensors:
            weights_path = os.path.join(save_dir, "model.safetensors")
            save_safetensors(self.model.state_dict(), weights_path)
        else:
            self.model.save_pretrained(save_dir)
