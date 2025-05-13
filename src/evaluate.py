import argparse
import yaml
from src.data_loader import load_data
from src.model import MedMTModel
import sacrebleu
from tqdm import tqdm

def evaluate(config):
    df = load_data(config['train_data'], is_train=True)
    model = MedMTModel(config['model_save_path'])
    srcs = df['source'].tolist()
    tgts = df['target'].tolist()
    preds = []
    for src in tqdm(srcs, desc="Evaluating"):
        pred = model.translate([src])[0]
        preds.append(pred)
    bleu = sacrebleu.corpus_bleu(preds, [tgts], smooth_method='exp')
    print(f"BLEU: {bleu.score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    evaluate(config)
