# -*- coding: utf-8 -*-
"""
Generate synthetic medical dialogue dataset for training (Chinese→Thai)

This dataset is generated using DeepSeek AI (https://deepseek.ai/)
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-SA-NC 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/

Usage:
    bash$ python src/generate_dataset.py --output data/synthetic_train.csv --n_samples 10000
    bash$ python src/generate_dataset.py --output data/qa_train.csv --n_samples 10000 --mode qa

Dependencies:
    pip install pandas python-dotenv openai tqdm
    For 'arrow' or 'parquet' format, also install: pip install pyarrow
"""
import sys
import os

# Add project root to sys.path to allow absolute imports from src
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import pandas as pd
import random
import re
from openai import OpenAI
from dotenv import load_dotenv
import time
from tqdm import tqdm as base_tqdm
import datetime
import string
import random as pyrandom
import logging
import concurrent.futures

# Import components
from src.components.banner import print_ascii_banner
from src.components.file_utils import save_dataset, recommend_format
from src.components.dataset_generators import (
    generate_qa_dataset,
    generate_deepseek_medical_dialogue
)
from src.components.display_utils import LoadingSpinner, print_section_header
from src.components.multiturn_generators import generate_multiturn_dialogues
from src.components.quality_assessment import run_quality_assessment
from src.components.response_cache import cached_api_call

# Import templates
try:
    from src.dataset_templates import (
        MEDICAL_TOPICS,
        DIALOGUE_SAMPLES,
        QA_SAMPLES,
        MEDICAL_REASONING_SAMPLES,
        SUMMARIZATION_SAMPLES,
        QA_PROMPT_TEMPLATE,
        REASONING_PROMPT_TEMPLATE,
        DIALOGUE_PROMPT_TEMPLATE,
        SUMMARIZATION_PROMPT_TEMPLATE
    )
except ImportError:
    print("[ERROR] Could not import templates.")
    print("Please ensure 'dataset_templates.py' exists in src/components/ directory")
    print("File path: src/components/dataset_templates.py")
    sys.exit(1)

# Setup error logger
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"generate_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def log_error(error_type, sample_idx, message):
    logging.error(f"{error_type} | Sample: {sample_idx} | {message}")

def main():
    print_ascii_banner()
    
    parser = argparse.ArgumentParser(description="Generate synthetic medical dialogue dataset")
    parser.add_argument("--output", type=str, help="Output file path", default="data/synthetic_train.csv")
    parser.add_argument("--n_samples", type=int, help="Number of samples to generate", default=100)
    parser.add_argument("--mode", type=str, choices=["dialogue", "qa", "reasoning", "summarization", "multiturn"],
                      help="Generation mode", default="dialogue")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--format", type=str, 
                      choices=["auto", "csv", "json", "jsonl", "txt", "arrow", "parquet"],
                      help="Output file format", default="auto")
    parser.add_argument("--max_workers", type=int,
                      help="Maximum number of concurrent workers for dialogue generation",
                      default=5)
    parser.add_argument("--assess_quality", action="store_true",
                      help="Run quality assessment on generated dataset")
    parser.add_argument("--use_cache", action="store_true",
                      help="Use cached API responses when available")
    
    args = parser.parse_args()
    
    if args.format == "auto":
        file_format = recommend_format(args.mode, args.output, args.n_samples)
        print(f'[INFO] Auto-selected output format: {file_format.upper()} (recommended for mode \'{args.mode}\' and n_samples={args.n_samples})')
    else:
        file_format = args.format
    
    print(f"[INFO] Mode: {args.mode} | Samples: {args.n_samples} | Output Suggestion: {args.output}")
    print(f"[INFO] Format: {file_format.upper()} | Max Workers (dialogue): {args.max_workers}")
    print(f"[INFO] Using DeepSeek API for generation. Log file: {LOG_FILE}")
    print_section_header("Dataset Generation Configuration")

    # Initialize loading spinner
    spinner = LoadingSpinner("Starting dataset generation...")
    spinner.start()
    
    try:
        # Setup phase
        spinner.message = "Setting up environment..."
        time.sleep(0.5)  # Short pause for visual feedback

        # Generation phase
        df = None
        start_time = time.time()
        
        # Pause spinner during generation to show progress bar
        spinner.stop()
        if args.mode == "dialogue":
            df = generate_deepseek_medical_dialogue(args.n_samples, args.seed, args.max_workers)
        elif args.mode == "qa":
            df = generate_qa_dataset(args.n_samples, args.seed)
        elif args.mode == "reasoning":
            df = generate_reasoning_dataset(args.n_samples, args.seed)
        elif args.mode == "summarization":
            df = generate_summarization_dataset(args.n_samples, args.seed)
        elif args.mode == "multiturn":
            df = generate_multiturn_dialogues(args.n_samples, args.seed, args.max_workers)
        else:
            print(f"[ERROR] Unsupported mode: {args.mode}")
            return 1

        # Run quality assessment if requested
        if args.assess_quality and not df.empty:
            print_section_header("Translation Quality Assessment")
            df = run_quality_assessment(df)

        # Save the generated dataset
        if not df.empty:
            # Saving phase
            spinner = LoadingSpinner("Saving dataset...")
            spinner.start()
            filepath, n_samples = save_dataset(df, args.output, file_format)
            
            # Complete
            end_time = time.time()
            duration = end_time - start_time
            spinner.stop(f"Dataset generation complete! Total samples: {n_samples} (Time: {duration:.1f}s)")
            
            print_section_header("Generation Summary")
            print(f"📊 Generated samples: {n_samples}")
            print(f"💾 Output file: {filepath}")
            print(f"⏱️ Processing time: {duration:.1f} seconds")
            
            if args.assess_quality:
                print_section_header("Quality Assessment")
                quality_report = run_quality_assessment(df)
                print(quality_report)
        else:
            print("[ERROR] No data was generated. Please check the logs for details.")
            return 1
        
        print(f"[INFO] Dataset is licensed under CC BY-SA-NC 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)")
        print(f"[INFO] Log file saved to: {LOG_FILE}")
        return 0
        
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        log_error("UnexpectedError", 0, str(e))
        return 1

if __name__ == "__main__":
    sys.exit(main())
