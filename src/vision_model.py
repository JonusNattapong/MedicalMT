# vision_model.py
"""
Multimodal Vision+LLM Model: ZombitxLM + CLIP-ViT (LLaVA/MiniGPT-4 style)
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor

class VisionLanguageModel(nn.Module):
    def __init__(self, llm_path, vision_encoder_name="openai/clip-vit-large-patch14"):
        super().__init__()
        # 1. Image encoder (CLIP)
        self.vision_encoder = CLIPModel.from_pretrained(vision_encoder_name)
        self.vision_processor = CLIPProcessor.from_pretrained(vision_encoder_name)
        # 2. LLM (ZombitxLM)
        self.llm_config = AutoConfig.from_pretrained(llm_path)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, config=self.llm_config)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        # 3. Projector: map vision embedding to LLM hidden size
        vision_dim = self.vision_encoder.config.projection_dim
        llm_dim = self.llm_config.hidden_size
        self.projector = nn.Linear(vision_dim, llm_dim)

    def forward(self, images, texts):
        # images: list of PIL.Image, texts: list of str
        device = next(self.parameters()).device
        img_inputs = self.vision_processor(images=images, return_tensors="pt").to(device)
        img_emb = self.vision_encoder.get_image_features(**img_inputs)
        img_emb_proj = self.projector(img_emb)
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        text_embeds = self.llm.get_input_embeddings()(tokens["input_ids"])
        # Insert image embedding as first token for each sample
        img_emb_proj = img_emb_proj.unsqueeze(1)
        input_embeds = torch.cat([img_emb_proj, text_embeds], dim=1)
        # Adjust attention mask
        attention_mask = torch.cat([
            torch.ones((tokens["input_ids"].shape[0], 1), dtype=tokens["attention_mask"].dtype, device=device),
            tokens["attention_mask"]
        ], dim=1)
        outputs = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=tokens["input_ids"])
        return outputs

    def generate(self, image, prompt, **gen_kwargs):
        device = next(self.parameters()).device
        img_inputs = self.vision_processor(images=image, return_tensors="pt").to(device)
        img_emb = self.vision_encoder.get_image_features(**img_inputs)
        img_emb_proj = self.projector(img_emb)
        tokens = self.tokenizer(prompt, return_tensors="pt").to(device)
        text_embeds = self.llm.get_input_embeddings()(tokens.input_ids)
        input_embeds = torch.cat([img_emb_proj.unsqueeze(1), text_embeds], dim=1)
        if hasattr(self.llm, "generate"):
            return self.llm.generate(inputs_embeds=input_embeds, **gen_kwargs)
        else:
            return self.forward([image], [prompt])
