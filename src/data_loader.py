import pandas as pd
import random
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder, create_repo, Repository
import os
from typing import Optional

def load_data(path: str, is_train: bool = True):
    df = pd.read_csv(path)
    if is_train:
        assert set(['context', 'source', 'target']).issubset(df.columns)
    else:
        assert set(['context', 'source']).issubset(df.columns)
    return df

def generate_synthetic_medical_dialogue(n_samples=1000, seed=42):
    random.seed(seed)
    data = []
    for i in range(n_samples):
        context = f"医生：你哪里不舒服？\\n病人：我头疼。"
        source = random.choice(["你吃过药了吗？", "你要休息一下。", "你需要做检查。", "你发烧了吗？"])
        target = random.choice(["คุณกินยาแล้วหรือยัง?", "คุณควรพักผ่อนสักหน่อย", "คุณต้องตรวจร่างกาย", "คุณมีไข้ไหม?"])
        data.append({"context": context, "source": source, "target": target})
    df = pd.DataFrame(data)
    return df

def save_dataset_to_hub(df, repo_name, split_name="train", private=False):
    api = HfApi()
    token = HfFolder.get_token()
    if token is None:
        raise RuntimeError("Please login to Hugging Face CLI: huggingface-cli login")
    repo_url = create_repo(repo_name, token=token, private=private, exist_ok=True)
    local_dir = f"./tmp_{repo_name.replace('/', '_')}"
    os.makedirs(local_dir, exist_ok=True)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(local_dir)
    repo = Repository(local_dir=local_dir, clone_from=repo_url.repo_url, use_auth_token=token)
    repo.git_add()
    repo.git_commit("Add synthetic dataset")
    repo.git_push()
    print(f"Dataset pushed to Hugging Face Hub: {repo_url.repo_url}")

if __name__ == "__main__":
    df = generate_synthetic_medical_dialogue(n_samples=1000)
    df.to_csv("data/synthetic_train.csv", index=False)
    # Uncomment to push to Hugging Face Hub
    # save_dataset_to_hub(df, repo_name="your-username/medmt-synthetic-dataset")
