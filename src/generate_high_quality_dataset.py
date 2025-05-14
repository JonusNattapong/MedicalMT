# -*- coding: utf-8 -*-
"""
High-quality dataset generation for Medical Machine Translation (Chinese → Thai)

This script combines multiple data generation approaches:
1. DeepSeek API for high-quality generation (if API key is available)
2. HuggingFace models for open-source alternative
3. Template-based generation for variety
4. Quality validation and filtering

Usage:
    python src/generate_high_quality_dataset.py --output data/high_quality_dataset 
                                              --samples 5000 --mode mixed
                                              --use_deepseek True --use_huggingface True
                                              --quality_filter 0.7
"""

import os
import sys
import time
import argparse
import random
import logging
import datetime
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project components
try:
    from src.components.banner import print_ascii_banner
    from src.components.file_utils import save_dataset, recommend_format, get_datasets_dir
    from src.components.dataset_generators import (
        generate_qa_dataset,
        generate_deepseek_medical_dialogue,
        generate_reasoning_dataset,
        generate_summarization_dataset
    )
    from src.components.display_utils import LoadingSpinner, print_section_header, ProgressBar
    from src.components.multiturn_generators import generate_multiturn_dialogues
    from src.components.quality_assessment import run_quality_assessment, assess_translations_with_model
    from src.components.response_cache import cached_api_call
    from src.components.metadata_utils import add_metadata
    from src.components.text_utils import extract_thai_translation

    # Import templates
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
except ImportError as e:
    print(f"[ERROR] Failed to import required modules: {e}")
    print("Please ensure all project components are available")
    sys.exit(1)

# Setup logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"generate_hq_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def log_error(error_type: str, sample_idx: int, message: str) -> None:
    """Log an error to both the logger and console"""
    error_msg = f"{error_type} | Sample: {sample_idx} | {message}"
    logger.error(error_msg)
    print(f"[ERROR] {error_msg}")

# HuggingFace model setup
MODEL_NAME = "Helsinki-NLP/opus-mt-zh-th"
MAX_LENGTH = 512

def initialize_huggingface_translator() -> Optional[pipeline]:
    """
    Initialize the HuggingFace translator model
    
    Returns:
        Translation pipeline if successful, None otherwise
    """
    try:
        print(f"[INFO] Initializing Hugging Face translator: {MODEL_NAME}")
        translator = pipeline(
            task="translation", 
            model=MODEL_NAME,
            tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
            max_length=MAX_LENGTH,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        print(f"[SUCCESS] Translator initialized: {MODEL_NAME}")
        print(f"[INFO] Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        return translator
    except Exception as e:
        print(f"[ERROR] Failed to initialize translator: {e}")
        logger.error(f"Translator initialization failed: {e}")
        return None

def initialize_deepseek_client() -> Optional[OpenAI]:
    """
    Initialize the DeepSeek API client
    
    Returns:
        OpenAI client if API key is available, None otherwise
    """
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("[WARNING] DEEPSEEK_API_KEY not found in environment variables or .env file")
        logger.warning("DeepSeek API key not found")
        return None
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        logger.info("DeepSeek client initialized")
        return client
    except Exception as e:
        print(f"[ERROR] Failed to initialize DeepSeek client: {e}")
        logger.error(f"DeepSeek client initialization failed: {e}")
        return None

def translate_with_huggingface(texts: Union[str, List[str]], 
                              translator: pipeline) -> Union[str, List[str]]:
    """
    Translate text using HuggingFace model
    
    Args:
        texts: Single text or list of texts to translate
        translator: HuggingFace translation pipeline
        
    Returns:
        Translated text(s)
    """
    if translator is None:
        log_error("TranslationError", -1, "Translator not initialized.")
        return "Translation unavailable" if isinstance(texts, str) else ["Translation unavailable"] * len(texts)
    
    try:
        # Handle single text
        if isinstance(texts, str):
            if not texts.strip():
                return ""
                
            translation_result = translator(texts)
            return translation_result[0]['translation_text']
            
        # Handle batch of texts
        elif isinstance(texts, list):
            if not texts:
                return []
                
            results = []
            batch_size = 8  # Process in small batches to avoid memory issues
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch = [txt for txt in batch if txt.strip()]  # Filter empty texts
                
                if not batch:
                    results.extend([""] * len(texts[i:i+batch_size]))
                    continue
                    
                batch_results = translator(batch)
                batch_translations = [res['translation_text'] for res in batch_results]
                
                # Add empty strings for any filtered texts
                j = 0
                for txt in texts[i:i+batch_size]:
                    if txt.strip():
                        results.append(batch_translations[j])
                        j += 1
                    else:
                        results.append("")
                        
            return results
            
        else:
            log_error("TranslationError", -1, f"Unsupported input type: {type(texts)}")
            return "Translation unavailable"
            
    except Exception as e:
        log_error("TranslationError", -1, f"Error during translation: {str(e)}")
        if isinstance(texts, str):
            return f"Translation error: {str(e)}"
        else:
            return [f"Translation error: {str(e)}"] * len(texts)

def generate_dialogue_dataset_hf(n_samples: int = 100, 
                               translator: pipeline = None,
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate dialogue dataset using HuggingFace model
    
    Args:
        n_samples: Number of samples to generate
        translator: HuggingFace translation pipeline
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with generated samples
    """
    if not translator:
        print("[ERROR] Translator not available for dialogue generation.")
        return pd.DataFrame()
    
    random.seed(seed)
    print(f"[INFO] Creating prompt combinations for {n_samples} dialogue samples...")
    
    all_prompt_combinations = []
    for topic in MEDICAL_TOPICS:
        for sample in DIALOGUE_SAMPLES:
            all_prompt_combinations.append((topic, sample))
    
    if not all_prompt_combinations:
        print("[ERROR] No dialogue samples or medical topics defined.")
        return pd.DataFrame()
    
    selected_combinations = []
    if n_samples <= len(all_prompt_combinations):
        selected_combinations = random.sample(all_prompt_combinations, n_samples)
    else:
        selected_combinations.extend(all_prompt_combinations)
        remaining_samples = n_samples - len(all_prompt_combinations)
        for i in range(remaining_samples):
            selected_combinations.append(all_prompt_combinations[i % len(all_prompt_combinations)])
    
    # Prepare for translation
    tasks_args = []
    for i, (topic, sample) in enumerate(selected_combinations):
        context = sample.get("context", "")
        source_text = sample.get("source", "")
        tasks_args.append((i, topic, {"context": context, "source": source_text}, translator, log_error))
    
    print(f"[INFO] Translating {len(tasks_args)} dialogue samples...")
    
    # Batch translation for efficiency
    source_texts_batch = [task[2]["source"] for task in tasks_args]
    translated_texts_batch = translate_with_huggingface(source_texts_batch, translator)
    
    data = []
    for i, task_arg_tuple in enumerate(tqdm(tasks_args, total=len(tasks_args), desc="Generating Dialogue Samples")):
        idx, topic, sample_data, _, _ = task_arg_tuple
        target_text = translated_texts_batch[i]
        
        if "Error:" in target_text or "Translation unavailable" in target_text:
            log_error("TranslationFailure", idx, f"Failed to translate. Source: {sample_data['source']}, Error: {target_text}")
            target_text = sample_data.get("target", target_text)
            
        data.append({
            "context": sample_data["context"], 
            "source": sample_data["source"], 
            "target": target_text,
            "topic_zh": topic["zh"],
            "topic_th": topic["th"],
            "topic_desc": topic["desc"]
        })
    
    df = pd.DataFrame(data)
    df = add_metadata(df)
    
    return df

def generate_qa_dataset_hf(n_samples: int = 100, 
                         translator: pipeline = None,
                         seed: int = 42) -> pd.DataFrame:
    """
    Generate QA dataset using HuggingFace model
    
    Args:
        n_samples: Number of samples to generate
        translator: HuggingFace translation pipeline
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with generated samples
    """
    if not translator:
        print("[ERROR] Translator not available for QA generation.")
        return pd.DataFrame()
    
    data = []
    random.seed(seed)
    
    if not QA_SAMPLES:
        print("[ERROR] No QA_SAMPLES defined.")
        return pd.DataFrame()
    
    for i in tqdm(range(n_samples), desc="Generating QA Samples"):
        sample = QA_SAMPLES[i % len(QA_SAMPLES)]
        question_zh = sample["question"]
        answer_zh = sample["answer"]
        
        try:
            question_th = translate_with_huggingface(question_zh, translator)
            answer_th = translate_with_huggingface(answer_zh, translator)
            
            if "Error:" in question_th or "Translation unavailable" in question_th:
                log_error("TranslationFailure", i, f"Question translation failed: {question_zh} -> {question_th}")
                question_th = sample.get("q_th", question_th)
                
            if "Error:" in answer_th or "Translation unavailable" in answer_th:
                log_error("TranslationFailure", i, f"Answer translation failed: {answer_zh} -> {answer_th}")
                answer_th = sample.get("a_th", answer_th)
                
        except Exception as e:
            log_error("HF_API_Error_QA", i, str(e))
            question_th = sample.get("q_th", f"Error: {e}")
            answer_th = sample.get("a_th", f"Error: {e}")
        
        data.append({
            "context": sample["context"],
            "question_zh": question_zh,
            "answer_zh": answer_zh,
            "question_th": question_th,
            "answer_th": answer_th
        })
        
    df = pd.DataFrame(data)
    df = add_metadata(df)
    
    return df

def generate_reasoning_dataset_hf(n_samples: int = 100, 
                                translator: pipeline = None,
                                seed: int = 42) -> pd.DataFrame:
    """
    Generate medical reasoning QA dataset using HuggingFace model
    
    Args:
        n_samples: Number of samples to generate
        translator: HuggingFace translation pipeline
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with generated samples
    """
    if not translator:
        print("[ERROR] Translator not available for reasoning generation.")
        return pd.DataFrame()
    
    data = []
    random.seed(seed)
    
    if not MEDICAL_REASONING_SAMPLES:
        print("[ERROR] No MEDICAL_REASONING_SAMPLES defined.")
        return pd.DataFrame()
    
    for i in tqdm(range(n_samples), desc="Generating Reasoning Samples"):
        sample = MEDICAL_REASONING_SAMPLES[i % len(MEDICAL_REASONING_SAMPLES)]
        question_zh = sample["question"]
        answer_zh = sample["answer"]
        
        try:
            question_th = translate_with_huggingface(question_zh, translator)
            answer_th = translate_with_huggingface(answer_zh, translator)
            
            if "Error:" in question_th or "Translation unavailable" in question_th:
                log_error("TranslationFailure", i, f"Question translation failed: {question_zh} -> {question_th}")
                question_th = sample.get("q_th", question_th)
                
            if "Error:" in answer_th or "Translation unavailable" in answer_th:
                log_error("TranslationFailure", i, f"Answer translation failed: {answer_zh} -> {answer_th}")
                answer_th = sample.get("a_th", answer_th)
                
        except Exception as e:
            log_error("HF_API_Error_Reasoning", i, str(e))
            question_th = sample.get("q_th", f"Error: {e}")
            answer_th = sample.get("a_th", f"Error: {e}")
            
        data.append({
            "context": sample["context"],
            "question_zh": question_zh,
            "answer_zh": answer_zh,
            "question_th": question_th,
            "answer_th": answer_th
        })
        
    df = pd.DataFrame(data)
    df = add_metadata(df)
    
    return df

def generate_summarization_dataset_hf(n_samples: int = 100, 
                                    translator: pipeline = None,
                                    seed: int = 42) -> pd.DataFrame:
    """
    Generate summarization dataset using HuggingFace model
    
    Args:
        n_samples: Number of samples to generate
        translator: HuggingFace translation pipeline
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with generated samples
    """
    if not translator:
        print("[ERROR] Translator not available for summarization generation.")
        return pd.DataFrame()
    
    data = []
    random.seed(seed)
    
    if not SUMMARIZATION_SAMPLES:
        print("[ERROR] No SUMMARIZATION_SAMPLES defined.")
        return pd.DataFrame()
    
    for i in tqdm(range(n_samples), desc="Generating Summarization Samples"):
        sample = SUMMARIZATION_SAMPLES[i % len(SUMMARIZATION_SAMPLES)]
        summary_zh = sample["summary_zh"]
        
        try:
            summary_th = translate_with_huggingface(summary_zh, translator)
            
            if "Error:" in summary_th or "Translation unavailable" in summary_th:
                log_error("TranslationFailure", i, f"Summary translation failed: {summary_zh} -> {summary_th}")
                summary_th = sample.get("summary_th", summary_th)
                
        except Exception as e:
            log_error("HF_API_Error_Summarization", i, str(e))
            summary_th = sample.get("summary_th", f"Error: {e}")
            
        data.append({
            "context": sample["context"],
            "summary_zh": summary_zh,
            "summary_th": summary_th
        })
        
    df = pd.DataFrame(data)
    df = add_metadata(df)
    
    return df

def filter_low_quality_samples(df: pd.DataFrame, 
                              quality_threshold: float = 0.7,
                              use_deepseek: bool = True) -> pd.DataFrame:
    """
    Filter out low-quality samples based on heuristics or model assessment
    
    Args:
        df: DataFrame with samples to filter
        quality_threshold: Threshold for minimum quality score (0.0-1.0)
        use_deepseek: Whether to use DeepSeek API for quality assessment
        
    Returns:
        Filtered DataFrame
    """
    print(f"[INFO] Filtering low-quality samples (threshold: {quality_threshold})")
    
    if df.empty:
        return df
    
    initial_count = len(df)
    
    # Basic heuristic filtering
    # 1. Remove empty translations
    if 'target' in df.columns:
        df = df[df['target'].str.strip().str.len() > 0]
    elif 'answer_th' in df.columns:
        df = df[df['answer_th'].str.strip().str.len() > 0]
    elif 'summary_th' in df.columns:
        df = df[df['summary_th'].str.strip().str.len() > 0]
    
    # 2. Remove very short translations (less than 5 characters)
    if 'target' in df.columns:
        df = df[df['target'].str.strip().str.len() >= 5]
    elif 'answer_th' in df.columns:
        df = df[df['answer_th'].str.strip().str.len() >= 5]
    elif 'summary_th' in df.columns:
        df = df[df['summary_th'].str.strip().str.len() >= 5]
    
    # 3. Check for translation failures
    for col in df.columns:
        if df[col].dtype == object:  # String columns
            mask = ~df[col].str.contains('Error:|Translation unavailable', na=False)
            df = df[mask]
    
    # 4. Use DeepSeek for quality assessment (if enabled)
    if use_deepseek:
        try:
            print("[INFO] Running DeepSeek quality assessment...")
            df = assess_translations_with_model(df, sample_ratio=0.2, min_samples=10)
            
            # Filter based on quality score
            if 'qe_score' in df.columns:
                normalized_score = df['qe_score'] / 10.0  # Convert 1-10 scale to 0-1
                df = df[normalized_score >= quality_threshold]
            
        except Exception as e:
            print(f"[WARNING] DeepSeek quality assessment failed: {e}")
            logger.warning(f"DeepSeek quality assessment failed: {e}")
    
    filtered_count = len(df)
    removed_count = initial_count - filtered_count
    removal_percentage = (removed_count / initial_count * 100) if initial_count > 0 else 0
    
    print(f"[INFO] Removed {removed_count} samples ({removal_percentage:.1f}%) due to quality issues")
    logger.info(f"Quality filtering removed {removed_count}/{initial_count} samples ({removal_percentage:.1f}%)")
    
    return df

def combine_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple datasets into one
    
    Args:
        dfs: List of DataFrames to combine
        
    Returns:
        Combined DataFrame
    """
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    
    if not valid_dfs:
        print("[WARNING] No valid datasets to combine")
        return pd.DataFrame()
    
    # Check if we have datasets with different structures
    is_translation = all('source' in df.columns and 'target' in df.columns for df in valid_dfs)
    is_qa = all('question_zh' in df.columns and 'answer_th' in df.columns for df in valid_dfs)
    is_summary = all('summary_zh' in df.columns and 'summary_th' in df.columns for df in valid_dfs)
    
    # If all datasets have the same structure, simply concatenate them
    if is_translation or is_qa or is_summary:
        df = pd.concat(valid_dfs, ignore_index=True)
        print(f"[INFO] Combined {len(valid_dfs)} datasets with {len(df)} total samples")
        return df
    
    # Handle different structures by converting to common format
    print("[INFO] Combining datasets with different structures...")
    
    combined_df = pd.DataFrame(columns=['source', 'target', 'context', 'task'])
    
    for df in valid_dfs:
        if df.empty:
            continue
            
        temp_df = pd.DataFrame(columns=['source', 'target', 'context', 'task'])
        
        if 'source' in df.columns and 'target' in df.columns:
            # Translation dataset
            temp_df['source'] = df['source']
            temp_df['target'] = df['target']
            temp_df['context'] = df['context'] if 'context' in df.columns else ''
            temp_df['task'] = 'translation'
            
        elif 'question_zh' in df.columns and 'answer_th' in df.columns:
            # QA dataset
            temp_df['source'] = df['question_zh']
            temp_df['target'] = df['answer_th']
            temp_df['context'] = df['context'] if 'context' in df.columns else ''
            temp_df['task'] = 'qa'
            
        elif 'summary_zh' in df.columns and 'summary_th' in df.columns:
            # Summarization dataset
            temp_df['source'] = df['context'] if 'context' in df.columns else ''
            temp_df['target'] = df['summary_th']
            temp_df['context'] = df['summary_zh']
            temp_df['task'] = 'summarization'
            
        # Combine with already converted datasets
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
    
    print(f"[INFO] Combined datasets with {len(combined_df)} total samples")
    return combined_df

def generate_high_quality_dataset(
    output_dir: str,
    n_samples: int = 1000,
    modes: List[str] = ["translation", "qa", "reasoning", "summarization"],
    use_deepseek: bool = True,
    use_huggingface: bool = True,
    quality_threshold: float = 0.7,
    max_workers: int = 5,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate high-quality dataset for various modes
    
    Args:
        output_dir: Directory to save the generated datasets
        n_samples: Number of samples to generate per mode
        modes: List of modes to generate datasets for
        use_deepseek: Whether to use DeepSeek API
        use_huggingface: Whether to use HuggingFace models
        quality_threshold: Threshold for minimum quality score
        max_workers: Maximum number of concurrent workers
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping modes to generated DataFrames
    """
    print_ascii_banner()
    print(f"[INFO] Generating high-quality dataset with {n_samples} samples per mode")
    logger.info(f"Dataset generation started with n_samples={n_samples}, modes={modes}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize clients
    deepseek_client = None
    hf_translator = None
    
    if use_deepseek:
        deepseek_client = initialize_deepseek_client()
        if not deepseek_client:
            print("[WARNING] DeepSeek client initialization failed; falling back to HuggingFace")
            use_huggingface = True
    
    if use_huggingface:
        hf_translator = initialize_huggingface_translator()
        if not hf_translator:
            print("[WARNING] HuggingFace translator initialization failed")
            if not deepseek_client:
                print("[ERROR] Both DeepSeek and HuggingFace initialization failed")
                logger.error("Both generation methods unavailable")
                return {}
    
    # Generate datasets for each mode
    datasets = {}
    
    for mode in modes:
        print_section_header(f"Generating {mode.upper()} dataset")
        logger.info(f"Starting generation for mode: {mode}")
        
        ds_deepseek = None
        ds_huggingface = None
        
        # Distribute samples between methods
        deepseek_samples = (n_samples // 2) if (use_deepseek and use_huggingface) else n_samples
        huggingface_samples = n_samples - deepseek_samples if use_huggingface else 0
        
        # Generate with DeepSeek
        if use_deepseek and deepseek_client:
            print(f"[INFO] Generating {deepseek_samples} samples using DeepSeek API")
            
            try:
                if mode == "translation" or mode == "dialogue":
                    ds_deepseek = generate_deepseek_medical_dialogue(
                        n_samples=deepseek_samples,
                        seed=seed,
                        max_workers=max_workers
                    )
                elif mode == "qa":
                    ds_deepseek = generate_qa_dataset(
                        n_samples=deepseek_samples,
                        seed=seed
                    )
                elif mode == "reasoning":
                    ds_deepseek = generate_reasoning_dataset(
                        n_samples=deepseek_samples,
                        seed=seed
                    )
                elif mode == "summarization":
                    ds_deepseek = generate_summarization_dataset(
                        n_samples=deepseek_samples,
                        seed=seed
                    )
                else:
                    print(f"[WARNING] Unsupported mode for DeepSeek: {mode}")
                    
            except Exception as e:
                print(f"[ERROR] DeepSeek generation failed for {mode}: {e}")
                logger.error(f"DeepSeek generation failed for {mode}: {e}")
        
        # Generate with HuggingFace
        if use_huggingface and hf_translator:
            print(f"[INFO] Generating {huggingface_samples} samples using HuggingFace model")
            
            try:
                if mode == "translation" or mode == "dialogue":
                    ds_huggingface = generate_dialogue_dataset_hf(
                        n_samples=huggingface_samples,
                        translator=hf_translator,
                        seed=seed+1  # Use different seed to ensure variety
                    )
                elif mode == "qa":
                    ds_huggingface = generate_qa_dataset_hf(
                        n_samples=huggingface_samples,
                        translator=hf_translator,
                        seed=seed+1
                    )
                elif mode == "reasoning":
                    ds_huggingface = generate_reasoning_dataset_hf(
                        n_samples=huggingface_samples,
                        translator=hf_translator,
                        seed=seed+1
                    )
                elif mode == "summarization":
                    ds_huggingface = generate_summarization_dataset_hf(
                        n_samples=huggingface_samples,
                        translator=hf_translator,
                        seed=seed+1
                    )
                else:
                    print(f"[WARNING] Unsupported mode for HuggingFace: {mode}")
                    
            except Exception as e:
                print(f"[ERROR] HuggingFace generation failed for {mode}: {e}")
                logger.error(f"HuggingFace generation failed for {mode}: {e}")
        
        # Combine and filter datasets
        datasets_to_combine = [ds for ds in [ds_deepseek, ds_huggingface] if ds is not None and not ds.empty]
        
        if not datasets_to_combine:
            print(f"[WARNING] No successful data generation for {mode}")
            logger.warning(f"No data generated for {mode}")
            continue
            
        combined_df = combine_datasets(datasets_to_combine)
        
        if combined_df.empty:
            print(f"[WARNING] Combined dataset for {mode} is empty")
            logger.warning(f"Empty combined dataset for {mode}")
            continue
            
        # Filter by quality
        filtered_df = filter_low_quality_samples(
            combined_df, 
            quality_threshold=quality_threshold,
            use_deepseek=(use_deepseek and deepseek_client is not None)
        )
        
        if filtered_df.empty:
            print(f"[WARNING] Filtered dataset for {mode} is empty")
            logger.warning(f"Empty filtered dataset for {mode}")
            continue
            
        # Store dataset
        datasets[mode] = filtered_df
        
        # Save to disk
        output_file = os.path.join(output_dir, f"{mode}_dataset.csv")
        filtered_df.to_csv(output_file, index=False)
        print(f"[SUCCESS] Saved {len(filtered_df)} samples to {output_file}")
        logger.info(f"Saved {len(filtered_df)} samples to {output_file}")
    
    # Generate combined dataset if multiple modes were used
    if len(datasets) > 1 and "mixed" in modes:
        print_section_header("Generating MIXED dataset")
        
        mixed_datasets = []
        total_samples = 0
        
        for mode, df in datasets.items():
            # Take equal portions from each dataset
            sample_size = min(len(df), n_samples // len(datasets))
            sampled_df = df.sample(sample_size, random_state=seed)
            mixed_datasets.append(sampled_df)
            total_samples += sample_size
            
        if mixed_datasets:
            mixed_df = combine_datasets(mixed_datasets)
            
            if not mixed_df.empty:
                output_file = os.path.join(output_dir, "mixed_dataset.csv")
                mixed_df.to_csv(output_file, index=False)
                print(f"[SUCCESS] Saved mixed dataset with {len(mixed_df)} samples to {output_file}")
                logger.info(f"Saved mixed dataset with {len(mixed_df)} samples to {output_file}")
                
                datasets["mixed"] = mixed_df
    
    print_section_header("Generation Summary")
    for mode, df in datasets.items():
        print(f"- {mode.upper()}: {len(df)} samples")
    
    return datasets

def generate_analysis(datasets: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Generate analysis reports and visualizations for the datasets
    
    Args:
        datasets: Dictionary mapping modes to DataFrames
        output_dir: Directory to save analysis outputs
    """
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    print_section_header("Generating Analysis")
    
    # Prepare summary data
    summary_data = []
    for mode, df in datasets.items():
        if df.empty:
            continue
            
        # Determine source and target columns based on mode
        if mode == "translation" or mode == "dialogue":
            source_col = "source"
            target_col = "target"
        elif mode == "qa" or mode == "reasoning":
            source_col = "question_zh"
            target_col = "answer_th"
        elif mode == "summarization":
            source_col = "summary_zh"
            target_col = "summary_th"
        elif mode == "mixed":
            # For mixed dataset, look at the columns
            if "source" in df.columns and "target" in df.columns:
                source_col = "source"
                target_col = "target"
            elif "question_zh" in df.columns:
                source_col = "question_zh"
                target_col = "answer_th"
            else:
                source_col = "summary_zh"
                target_col = "summary_th"
        else:
            # Skip if unknown mode
            continue
            
        # Skip if the columns don't exist
        if source_col not in df.columns or target_col not in df.columns:
            continue
            
        # Calculate statistics
        source_lengths = df[source_col].str.len()
        target_lengths = df[target_col].str.len()
        
        mode_summary = {
            "mode": mode,
            "samples": len(df),
            "avg_source_len": source_lengths.mean(),
            "avg_target_len": target_lengths.mean(),
            "min_source_len": source_lengths.min(),
            "max_source_len": source_lengths.max(),
            "min_target_len": target_lengths.min(),
            "max_target_len": target_lengths.max()
        }
        
        summary_data.append(mode_summary)
        
        # Generate visualizations
        plt.figure(figsize=(12, 6))
        
        # Source length distribution
        plt.subplot(1, 2, 1)
        sns.histplot(source_lengths, kde=True)
        plt.title(f"{mode.capitalize()} Source Length Distribution")
        plt.xlabel("Length (characters)")
        plt.ylabel("Count")
        
        # Target length distribution
        plt.subplot(1, 2, 2)
        sns.histplot(target_lengths, kde=True)
        plt.title(f"{mode.capitalize()} Target Length Distribution")
        plt.xlabel("Length (characters)")
        plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"{mode}_length_dist.png"))
        plt.close()
        
        # Generate quality score distribution if available
        if "qe_score" in df.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(df["qe_score"], kde=True, bins=10)
            plt.title(f"{mode.capitalize()} Quality Score Distribution")
            plt.xlabel("Quality Score (1-10)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"{mode}_quality_dist.png"))
            plt.close()
    
    # Generate summary report
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save as CSV
        summary_df.to_csv(os.path.join(analysis_dir, "dataset_summary.csv"), index=False)
        
        # Generate markdown report
        with open(os.path.join(analysis_dir, "dataset_report.md"), "w", encoding="utf-8") as f:
            f.write("# High-Quality Dataset Analysis Report\n\n")
            f.write("## Dataset Summary\n\n")
            
            f.write("| Mode | Samples | Avg Src Len | Avg Tgt Len | Min Src Len | Max Src Len | Min Tgt Len | Max Tgt Len |\n")
            f.write("|------|---------|------------|------------|------------|------------|------------|------------|\n")
            
            for _, row in summary_df.iterrows():
                f.write(f"| {row['mode']} | {row['samples']:,} | {row['avg_source_len']:.1f} | {row['avg_target_len']:.1f} | ")
                f.write(f"{row['min_source_len']} | {row['max_source_len']} | {row['min_target_len']} | {row['max_target_len']} |\n")
                
            f.write("\n\n## Dataset Visualizations\n\n")
            
            for mode in summary_df["mode"]:
                f.write(f"### {mode.capitalize()} Mode\n\n")
                f.write(f"![{mode.capitalize()} Length Distribution]({mode}_length_dist.png)\n\n")
                
                if os.path.exists(os.path.join(analysis_dir, f"{mode}_quality_dist.png")):
                    f.write(f"![{mode.capitalize()} Quality Distribution]({mode}_quality_dist.png)\n\n")
        
        print(f"[SUCCESS] Analysis report generated at {os.path.join(analysis_dir, 'dataset_report.md')}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="High-quality medical dataset generator")
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/high_quality",
        help="Output directory for datasets"
    )
    
    parser.add_argument(
        "--samples", 
        type=int, 
        default=1000,
        help="Number of samples to generate per mode"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="mixed",
        help="Generation mode(s) (comma-separated): translation,qa,reasoning,summarization,mixed"
    )
    
    parser.add_argument(
        "--use_deepseek", 
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Use DeepSeek API for generation (True/False)"
    )
    
    parser.add_argument(
        "--use_huggingface", 
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Use HuggingFace models for generation (True/False)"
    )
    
    parser.add_argument(
        "--quality_filter", 
        type=float, 
        default=0.7,
        help="Quality threshold for filtering (0.0-1.0)"
    )
    
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=5,
        help="Maximum number of concurrent workers"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Parse modes
    modes = [mode.strip() for mode in args.mode.split(",")]
    
    start_time = time.time()
    
    # Generate datasets
    datasets = generate_high_quality_dataset(
        output_dir=args.output,
        n_samples=args.samples,
        modes=modes,
        use_deepseek=args.use_deepseek,
        use_huggingface=args.use_huggingface,
        quality_threshold=args.quality_filter,
        max_workers=args.max_workers,
        seed=args.seed
    )
    
    # Generate analysis
    if datasets:
        generate_analysis(datasets, args.output)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print_section_header("Process Complete")
    print(f"Total time: {duration:.1f} seconds")
    print(f"Datasets saved to: {args.output}")
    print(f"Log file saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
