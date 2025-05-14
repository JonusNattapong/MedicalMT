# -*- coding: utf-8 -*-
"""
Script: generate_with_deepseek.py
สร้างข้อมูลใหม่ด้วย DeepSeek API (Medical MT)

Usage:
    python src/generate_with_deepseek.py --prompt_template "{context}\n{question}" --input_csv data/qa_train.csv --output_csv data/deepseek_generated.csv --n_samples 100

Environment:
    ต้องตั้งค่า DEEPSEEK_API_KEY ใน .env หรือ environment variable
"""
import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    logger.error("DEEPSEEK_API_KEY not found. Please set it in your .env file or environment.")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

# Default prompt template
DEFAULT_TEMPLATE = "บริบท: {context}\nคำถาม: {question}\nคำแปลภาษาไทย:"


def call_deepseek(prompt, model="deepseek-chat", max_tokens=256, temperature=0.7, top_p=0.95):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Generate data with DeepSeek API")
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file (must have columns context/question)')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file')
    parser.add_argument('--prompt_template', type=str, default=DEFAULT_TEMPLATE, help='Prompt template (use {context}, {question})')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--model', type=str, default="deepseek-chat", help='DeepSeek model name')
    parser.add_argument('--max_tokens', type=int, default=256, help='Max tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if 'context' not in df.columns or 'question' not in df.columns:
        logger.error('Input CSV must have columns: context, question')
        sys.exit(1)

    results = []
    for i, row in tqdm(df.iterrows(), total=min(args.n_samples, len(df)), desc="Generating"):
        if i >= args.n_samples:
            break
        prompt = args.prompt_template.format(context=row['context'], question=row['question'])
        answer = call_deepseek(prompt, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
        results.append({
            'context': row['context'],
            'question': row['question'],
            'deepseek_answer': answer
        })
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved {len(out_df)} samples to {args.output_csv}")

if __name__ == "__main__":
    main()
