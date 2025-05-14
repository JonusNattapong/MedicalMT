# -*- coding: utf-8 -*-
"""
Comprehensive dataset preparation pipeline for MedicalMT (Chinese → Thai)

This module provides tools for:
1. Data collection from multiple sources
2. Data preprocessing and cleaning
3. Quality validation
4. Dataset splitting (train/val/test)

Usage:
    python src/dataset_preparation.py --source_dirs data/raw --output_dir data/processed 
                                      --tasks translation,qa,summarization,reasoning
                                      --validation_split 0.1 --test_split 0.1
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
import logging
import glob
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.quality_assessment import assess_translations_with_model
from src.components.file_utils import get_datasets_dir, unique_dataset_filename, save_dataset

# Set up logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_COLUMN_MAPPINGS = {
    "translation": {
        "source_col": "source", 
        "target_col": "target", 
        "context_col": "context"
    },
    "qa": {
        "source_col": "question_zh", 
        "target_col": "answer_th", 
        "context_col": "context"
    },
    "summarization": {
        "source_col": "context", 
        "target_col": "summary_th", 
        "context_col": None
    },
    "reasoning": {
        "source_col": "question_zh", 
        "target_col": "answer_th", 
        "context_col": "context"
    }
}

def scan_data_sources(source_dirs: List[str]) -> Dict[str, List[str]]:
    """
    Scan provided directories for dataset files
    
    Args:
        source_dirs: List of directories to scan for datasets
        
    Returns:
        Dictionary mapping file extensions to lists of file paths
    """
    logger.info(f"Scanning directories: {source_dirs}")
    
    # Initialize a dictionary to hold files by extension
    files_by_ext = {}
    
    # Look for files in each source directory
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            logger.warning(f"Directory does not exist: {source_dir}")
            continue
            
        # Walk through directory recursively
        for root, _, files in os.walk(source_dir):
            for file in files:
                # Skip hidden files and directories
                if file.startswith('.'):
                    continue
                    
                # Get file extension
                _, ext = os.path.splitext(file)
                ext = ext.lower().lstrip('.')
                
                # Skip if not a common dataset format
                if ext not in ['csv', 'json', 'jsonl', 'parquet', 'arrow', 'xlsx']:
                    continue
                    
                # Add file to appropriate list
                if ext not in files_by_ext:
                    files_by_ext[ext] = []
                    
                files_by_ext[ext].append(os.path.join(root, file))
    
    # Log summary of found files
    total_files = sum(len(files) for files in files_by_ext.values())
    logger.info(f"Found {total_files} dataset files:")
    for ext, files in files_by_ext.items():
        logger.info(f"  - {ext.upper()}: {len(files)} files")
        
    return files_by_ext

def load_dataset_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load a dataset file into a pandas DataFrame
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        DataFrame containing the dataset, or None if loading fails
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower().lstrip('.')
    
    logger.info(f"Loading dataset from {file_path}")
    
    try:
        if ext == 'csv':
            # Try with different encodings in case of encoding issues
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='utf-8')
                
        elif ext == 'json':
            df = pd.read_json(file_path)
            
        elif ext == 'jsonl':
            df = pd.read_json(file_path, lines=True)
            
        elif ext in ['parquet', 'arrow']:
            df = pd.read_parquet(file_path)
            
        elif ext == 'xlsx':
            df = pd.read_excel(file_path)
            
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return None
            
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load dataset {file_path}: {str(e)}")
        return None

def infer_dataset_task(df: pd.DataFrame) -> str:
    """
    Infer the dataset task from its structure
    
    Args:
        df: DataFrame containing the dataset
        
    Returns:
        Task name: 'translation', 'qa', 'summarization', or 'reasoning'
    """
    columns = set(df.columns)
    
    # Check for QA dataset
    if {'question_zh', 'answer_zh', 'question_th', 'answer_th'}.issubset(columns):
        return 'qa'
        
    # Check for reasoning dataset
    if {'question_zh', 'answer_th', 'context'}.issubset(columns):
        return 'reasoning'
        
    # Check for summarization dataset
    if {'context', 'summary_zh', 'summary_th'}.issubset(columns):
        return 'summarization'
        
    # Default to translation dataset
    if {'source', 'target'}.issubset(columns):
        return 'translation'
        
    # Try to guess based on column names
    if 'question' in columns or 'answer' in columns:
        return 'qa'
    elif 'summary' in columns or 'abstract' in columns:
        return 'summarization'
        
    # Fall back to translation as the default
    return 'translation'

def clean_text(text: str) -> str:
    """
    Clean text data
    
    Args:
        text: The input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
        
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    return text

def detect_language(text: str) -> Optional[str]:
    """
    Detect if text is Chinese, Thai, or another language
    
    Args:
        text: The input text to analyze
        
    Returns:
        Language code ('zh', 'th', or None)
    """
    if not text or not isinstance(text, str):
        return None
        
    # Count Chinese characters
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    chinese_count = len(chinese_pattern.findall(text))
    
    # Count Thai characters
    thai_pattern = re.compile(r'[\u0e01-\u0e5b]')
    thai_count = len(thai_pattern.findall(text))
    
    # Determine language based on character count
    if chinese_count > thai_count and chinese_count > len(text) * 0.3:
        return 'zh'
    elif thai_count > chinese_count and thai_count > len(text) * 0.3:
        return 'th'
    else:
        return None

def validate_parallel_text(source: str, target: str) -> bool:
    """
    Validate if source and target texts form a valid translation pair
    
    Args:
        source: Source text (Chinese)
        target: Target text (Thai)
        
    Returns:
        True if the pair is valid, False otherwise
    """
    if not source or not target:
        return False
        
    # Detect languages
    source_lang = detect_language(source)
    target_lang = detect_language(target)
    
    # Check if languages match expected values
    if source_lang != 'zh' or target_lang != 'th':
        return False
        
    # Check for extreme length differences
    source_len = len(source)
    target_len = len(target)
    
    if source_len == 0 or target_len == 0:
        return False
        
    # Thai typically requires fewer characters than Chinese
    # But extreme differences might indicate incomplete translations
    ratio = source_len / target_len
    
    if ratio < 0.2 or ratio > 5.0:
        return False
        
    return True

def preprocess_dataset(df: pd.DataFrame, task: str, 
                       column_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Preprocess a dataset for the specified task
    
    Args:
        df: DataFrame containing the dataset
        task: Task name ('translation', 'qa', 'summarization', or 'reasoning')
        column_mapping: Optional mapping of column names
        
    Returns:
        Preprocessed DataFrame
    """
    # Use default column mapping if none provided
    if column_mapping is None:
        column_mapping = DEFAULT_COLUMN_MAPPINGS.get(task, DEFAULT_COLUMN_MAPPINGS['translation'])
    
    logger.info(f"Preprocessing dataset for task: {task}")
    logger.info(f"Initial dataset shape: {df.shape}")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Map the column names if necessary
    source_col = column_mapping.get('source_col')
    target_col = column_mapping.get('target_col')
    context_col = column_mapping.get('context_col')
    
    # Ensure required columns exist
    required_cols = [col for col in [source_col, target_col, context_col] if col is not None]
    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        
        # Try to infer column mapping if standard columns are missing
        if source_col in missing_cols and 'source' in df_clean.columns:
            df_clean[source_col] = df_clean['source']
            logger.info(f"Mapped 'source' to '{source_col}'")
            
        if target_col in missing_cols and 'target' in df_clean.columns:
            df_clean[target_col] = df_clean['target']
            logger.info(f"Mapped 'target' to '{target_col}'")
            
        if context_col in missing_cols and 'context' in df_clean.columns:
            df_clean[context_col] = df_clean['context']
            logger.info(f"Mapped 'context' to '{context_col}'")
            
        # If still missing required columns, we can't proceed
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        if missing_cols:
            logger.error(f"Still missing required columns after mapping: {missing_cols}")
            return pd.DataFrame()
    
    # Clean text in all columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).apply(clean_text)
    
    # Validate parallel text pairs
    if task == 'translation':
        valid_pairs = df_clean.apply(
            lambda row: validate_parallel_text(row[source_col], row[target_col]), 
            axis=1
        )
        df_clean = df_clean[valid_pairs]
        logger.info(f"Removed {(~valid_pairs).sum()} invalid translation pairs")
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=[source_col, target_col])
    logger.info(f"Removed duplicates. New shape: {df_clean.shape}")
    
    # Handle empty contexts if context column exists
    if context_col in df_clean.columns:
        df_clean[context_col] = df_clean[context_col].fillna("")
    
    return df_clean

def merge_datasets(dfs: List[pd.DataFrame], task: str) -> pd.DataFrame:
    """
    Merge multiple datasets for the same task
    
    Args:
        dfs: List of DataFrames to merge
        task: Task name
        
    Returns:
        Merged DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    
    logger.info(f"Merging {len(dfs)} datasets for task: {task}")
    
    # Get column mapping for this task
    column_mapping = DEFAULT_COLUMN_MAPPINGS.get(task, DEFAULT_COLUMN_MAPPINGS['translation'])
    source_col = column_mapping['source_col']
    target_col = column_mapping['target_col']
    context_col = column_mapping['context_col']
    
    # Ensure all dataframes have the required columns
    valid_dfs = []
    for i, df in enumerate(dfs):
        required_cols = [col for col in [source_col, target_col, context_col] if col is not None]
        if all(col in df.columns for col in required_cols):
            valid_dfs.append(df)
        else:
            logger.warning(f"Dataset {i} missing required columns. Skipping.")
    
    if not valid_dfs:
        logger.error(f"No valid datasets to merge for task: {task}")
        return pd.DataFrame()
    
    # Concatenate datasets
    merged_df = pd.concat(valid_dfs, ignore_index=True)
    
    # Remove duplicates
    merged_df = merged_df.drop_duplicates(subset=[source_col, target_col])
    
    logger.info(f"Merged dataset shape: {merged_df.shape}")
    return merged_df

def split_dataset(df: pd.DataFrame, val_ratio: float = 0.1, 
                  test_ratio: float = 0.1, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Split a dataset into training, validation, and test sets
    
    Args:
        df: DataFrame to split
        val_ratio: Proportion of data to use for validation
        test_ratio: Proportion of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of DataFrames for 'train', 'val', and 'test' sets
    """
    if df.empty:
        return {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
    
    logger.info(f"Splitting dataset (val: {val_ratio}, test: {test_ratio})")
    
    # Calculate actual proportions
    train_ratio = 1.0 - val_ratio - test_ratio
    
    # First split off the test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed
    )
    
    # Then split the train_val set into train and validation sets
    val_adjusted_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_adjusted_ratio, random_state=seed
    )
    
    # Log split sizes
    logger.info(f"Split sizes - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

def generate_analysis_report(datasets: Dict[str, Dict[str, pd.DataFrame]], 
                           output_dir: str) -> None:
    """
    Generate analysis report for prepared datasets
    
    Args:
        datasets: Dictionary of task -> split -> DataFrame
        output_dir: Directory to save analysis results
    """
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    all_stats = {}
    
    for task, splits in datasets.items():
        task_dir = os.path.join(analysis_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        
        # Get column mapping for this task
        column_mapping = DEFAULT_COLUMN_MAPPINGS.get(task, DEFAULT_COLUMN_MAPPINGS['translation'])
        source_col = column_mapping['source_col']
        target_col = column_mapping['target_col']
        
        task_stats = {}
        
        # Analyze each split
        for split_name, df in splits.items():
            if df.empty:
                continue
                
            split_stats = {
                'sample_count': len(df),
                'avg_source_length': df[source_col].str.len().mean(),
                'avg_target_length': df[target_col].str.len().mean(),
                'unique_source_count': df[source_col].nunique(),
                'unique_target_count': df[target_col].nunique()
            }
            
            # Generate visualizations
            plt.figure(figsize=(12, 6))
            
            # Source length distribution
            plt.subplot(1, 2, 1)
            sns.histplot(df[source_col].str.len(), kde=True)
            plt.title(f'{task.capitalize()} - {split_name.capitalize()} Source Length')
            plt.xlabel('Length (characters)')
            plt.ylabel('Count')
            
            # Target length distribution
            plt.subplot(1, 2, 2)
            sns.histplot(df[target_col].str.len(), kde=True)
            plt.title(f'{task.capitalize()} - {split_name.capitalize()} Target Length')
            plt.xlabel('Length (characters)')
            plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, f'{split_name}_length_dist.png'))
            plt.close()
            
            # Store stats
            task_stats[split_name] = split_stats
            
        all_stats[task] = task_stats
        
    # Save all stats as JSON
    with open(os.path.join(analysis_dir, 'dataset_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
        
    # Generate summary report
    with open(os.path.join(analysis_dir, 'summary_report.md'), 'w', encoding='utf-8') as f:
        f.write('# Dataset Preparation Summary Report\n\n')
        
        for task, task_stats in all_stats.items():
            f.write(f'## {task.capitalize()} Task\n\n')
            
            # Create table for this task
            f.write('| Split | Samples | Avg Source Len | Avg Target Len | Unique Sources | Unique Targets |\n')
            f.write('|-------|---------|----------------|----------------|----------------|----------------|\n')
            
            for split_name, stats in task_stats.items():
                f.write(f"| {split_name.capitalize()} | {stats['sample_count']:,} | ")
                f.write(f"{stats['avg_source_length']:.1f} | {stats['avg_target_length']:.1f} | ")
                f.write(f"{stats['unique_source_count']:,} | {stats['unique_target_count']:,} |\n")
                
            f.write('\n')
            
    logger.info(f"Analysis report generated at {analysis_dir}")

def run_pipeline(source_dirs: List[str], output_dir: str, tasks: List[str],
                val_split: float = 0.1, test_split: float = 0.1, 
                assess_quality: bool = False, seed: int = 42) -> None:
    """
    Run the complete dataset preparation pipeline
    
    Args:
        source_dirs: List of directories containing source datasets
        output_dir: Directory to save processed datasets
        tasks: List of tasks to prepare datasets for
        val_split: Proportion of data to use for validation
        test_split: Proportion of data to use for testing
        assess_quality: Whether to run quality assessment
        seed: Random seed for reproducibility
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Scan for data sources
    files_by_ext = scan_data_sources(source_dirs)
    
    if not any(files_by_ext.values()):
        logger.error("No dataset files found in the provided directories")
        return
    
    # Store loaded datasets by task
    datasets_by_task = {task: [] for task in tasks}
    
    # Load and preprocess datasets
    for ext, file_paths in files_by_ext.items():
        for file_path in file_paths:
            # Load dataset
            df = load_dataset_file(file_path)
            
            if df is None or df.empty:
                continue
                
            # Infer task if not specified in filename
            inferred_task = infer_dataset_task(df)
            
            # Skip if inferred task is not in requested tasks
            if inferred_task not in tasks:
                logger.info(f"Skipping {file_path} (task: {inferred_task}) as it's not in requested tasks")
                continue
                
            # Preprocess dataset
            preprocessed_df = preprocess_dataset(df, inferred_task)
            
            if not preprocessed_df.empty:
                datasets_by_task[inferred_task].append(preprocessed_df)
                logger.info(f"Added {len(preprocessed_df)} samples from {file_path} for {inferred_task} task")
    
    # Merge, split, and save datasets for each task
    final_datasets = {}
    
    for task in tasks:
        if not datasets_by_task[task]:
            logger.warning(f"No datasets found for task: {task}")
            continue
            
        # Merge datasets for this task
        merged_df = merge_datasets(datasets_by_task[task], task)
        
        if merged_df.empty:
            continue
            
        # Assess quality if requested
        if assess_quality:
            logger.info(f"Running quality assessment for {task} task")
            try:
                merged_df = assess_translations_with_model(merged_df, sample_ratio=0.1, min_samples=5)
                logger.info("Quality assessment complete")
            except Exception as e:
                logger.error(f"Quality assessment failed: {str(e)}")
        
        # Split dataset
        splits = split_dataset(merged_df, val_split, test_split, seed)
        
        # Save each split
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        
        for split_name, split_df in splits.items():
            if split_df.empty:
                continue
                
            output_path = os.path.join(task_dir, f"{split_name}.csv")
            split_df.to_csv(output_path, index=False)
            logger.info(f"Saved {split_name} split with {len(split_df)} samples to {output_path}")
        
        # Store for analysis
        final_datasets[task] = splits
    
    # Generate analysis report
    generate_analysis_report(final_datasets, output_dir)
    
    logger.info("Dataset preparation pipeline completed successfully")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Medical MT Dataset Preparation Pipeline")
    
    parser.add_argument(
        "--source_dirs", 
        type=str, 
        nargs='+',
        required=True,
        help="Directories containing source datasets"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed",
        help="Directory to save processed datasets"
    )
    
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="translation,qa,summarization,reasoning",
        help="Comma-separated list of tasks to prepare datasets for"
    )
    
    parser.add_argument(
        "--validation_split", 
        type=float, 
        default=0.1,
        help="Proportion of data to use for validation"
    )
    
    parser.add_argument(
        "--test_split", 
        type=float, 
        default=0.1,
        help="Proportion of data to use for testing"
    )
    
    parser.add_argument(
        "--assess_quality", 
        action="store_true",
        help="Run quality assessment on datasets"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Parse tasks
    tasks = [task.strip() for task in args.tasks.split(',')]
    
    # Run pipeline
    run_pipeline(
        source_dirs=args.source_dirs,
        output_dir=args.output_dir,
        tasks=tasks,
        val_split=args.validation_split,
        test_split=args.test_split,
        assess_quality=args.assess_quality,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
