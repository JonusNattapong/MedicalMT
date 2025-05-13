import argparse
import yaml
from src.data_loader import load_data
from src.model import MedMTModel
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from tqdm import tqdm
import os
from huggingface_hub import HfApi, HfFolder, Repository, create_repo

class MedDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.df.iloc[idx]['source'], self.df.iloc[idx]['target']

def train(config):
    df = load_data(config['train_data'], is_train=True)
    model = MedMTModel(config['pretrained_model'])
    optimizer = AdamW(model.model.parameters(), lr=config['learning_rate'])
    dataset = MedDataset(df)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    model.model.train()
    for epoch in range(config['epochs']):
        for src, tgt in tqdm(loader, desc=f"Epoch {epoch+1}"):
            inputs = model.tokenizer(list(src), return_tensors="pt", padding=True, truncation=True, max_length=config['max_seq_length']).to(model.device)
            labels = model.tokenizer(list(tgt), return_tensors="pt", padding=True, truncation=True, max_length=config['max_seq_length']).input_ids.to(model.device)
            outputs = model.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    # Save as .safetensors and normal format
    model.model.save_pretrained(config['model_save_path'])
    model.tokenizer.save_pretrained(config['model_save_path'])
    try:
        from safetensors.torch import save_file as save_safetensors
        import torch
        save_safetensors(model.model.state_dict(), os.path.join(config['model_save_path'], 'model.safetensors'))
        print(f"Model weights saved as .safetensors at {config['model_save_path']}")
    except ImportError:
        print("safetensors not installed, skipping .safetensors save.")

def push_model_to_hub(model_dir, repo_name, commit_message="Add model", private=False):
    api = HfApi()
    token = HfFolder.get_token()
    if token is None:
        raise RuntimeError("Please login to Hugging Face CLI: huggingface-cli login")
    repo_url = create_repo(repo_name, token=token, private=private, exist_ok=True)
    repo = Repository(local_dir=model_dir, clone_from=repo_url.repo_url, use_auth_token=token)
    repo.git_add()
    repo.git_commit(commit_message)
    repo.git_push()
    print(f"Model pushed to Hugging Face Hub: {repo_url.repo_url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--push_to_hub', action='store_true', help='Push model to Hugging Face Hub')
    parser.add_argument('--hf_repo', type=str, default=None, help='Hugging Face repo name (username/repo)')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    train(config)
    if args.push_to_hub and args.hf_repo:
        push_model_to_hub(config['model_save_path'], args.hf_repo)
