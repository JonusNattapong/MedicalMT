import argparse
import yaml
import pandas as pd
from src.data_loader import load_data
from src.model import MedMTModel
from tqdm import tqdm
import os

def inference(config, input_path, output_path):
    df = load_data(input_path, is_train=False)
    model = MedMTModel(config['model_save_path'])
    srcs = df['source'].tolist()
    preds = []
    for src in tqdm(srcs, desc="Translating"):
        pred = model.translate([src])[0]
        preds.append(pred)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame({'id': range(len(preds)), 'translation': preds}).to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    inference(config, args.input, args.output)
