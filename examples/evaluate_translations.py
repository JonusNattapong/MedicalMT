"""
Evaluate translation quality against a reference dataset.

This script compares translations from the XiaoduoAILab/XmodelLM1.5 model
against reference translations using BLEU, METEOR, and other metrics.
"""
import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
import nltk
from src.load_xiaoduo_model import load_xiaoduo_model

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('metrics/meteor')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_bleu(reference, candidate):
    """
    Compute BLEU score between reference and candidate translations.
    
    Args:
        reference (str): Reference translation
        candidate (str): Candidate translation to evaluate
        
    Returns:
        float: BLEU score
    """
    # Tokenize the strings into words
    reference_tokens = word_tokenize(reference.lower())
    candidate_tokens = word_tokenize(candidate.lower())
    
    # Compute BLEU score (using NLTK's implementation)
    # We use smoothing to handle cases where there are no n-gram matches
    weights = (0.25, 0.25, 0.25, 0.25)  # default weights for BLEU-4
    
    # Handle empty tokens by returning 0
    if not candidate_tokens:
        return 0.0
    
    score = sentence_bleu([reference_tokens], candidate_tokens, weights=weights, smoothing_function=None)
    
    return score

def compute_meteor(reference, candidate):
    """
    Compute METEOR score between reference and candidate translations.
    
    Args:
        reference (str): Reference translation
        candidate (str): Candidate translation to evaluate
        
    Returns:
        float: METEOR score
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        
        # Tokenize the strings into words
        reference_tokens = word_tokenize(reference.lower())
        candidate_tokens = word_tokenize(candidate.lower())
        
        # Compute METEOR score
        score = meteor_score([reference_tokens], candidate_tokens)
        
        return score
    except (ImportError, AttributeError):
        logger.warning("METEOR score calculation failed. Please ensure nltk is installed with the required data.")
        return 0.0

def compute_character_match_ratio(reference, candidate):
    """
    Compute the character match ratio between reference and candidate.
    This is a simple metric that measures the ratio of matching characters.
    
    Args:
        reference (str): Reference translation
        candidate (str): Candidate translation to evaluate
        
    Returns:
        float: Character match ratio (0-1)
    """
    # Convert to sets of characters
    ref_chars = set(reference)
    cand_chars = set(candidate)
    
    # Compute intersection and union
    intersection = ref_chars.intersection(cand_chars)
    union = ref_chars.union(cand_chars)
    
    # Compute Jaccard similarity
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

def evaluate_translations(dataset_path, output_path=None, num_samples=None):
    """
    Evaluate the XmodelLM1.5 model's translations against reference translations.
    
    Args:
        dataset_path (str): Path to the dataset CSV with 'source' and 'target' columns
        output_path (str, optional): Path to save evaluation results
        num_samples (int, optional): Number of samples to evaluate (if None, use all)
        
    Returns:
        dict: Dictionary with evaluation results
    """
    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        if 'source_text' in df.columns and 'target_text' in df.columns:
            source_col, target_col = 'source_text', 'target_text'
        elif 'zh' in df.columns and 'th' in df.columns:
            source_col, target_col = 'zh', 'th'
        else:
            raise ValueError(f"Dataset must have source and target columns. Found: {df.columns}")
            
        # Limit samples if specified
        if num_samples is not None:
            df = df.head(num_samples)
            
        logger.info(f"Loaded {len(df)} samples for evaluation")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None
    
    # Load the model
    logger.info("Loading XmodelLM1.5 model...")
    model = load_xiaoduo_model(use_instruct_version=True)
    
    if not model:
        logger.error("Failed to load the model.")
        return None
    
    # Custom prompt for medical dialogue translation
    system_prompt = (
        "You are a medical translator specialized in Chinese to Thai translation. "
        "Translate the provided Chinese medical dialogue to Thai accurately, "
        "preserving medical terminology and the natural conversational flow. "
    )
    
    # Translate and evaluate each sample
    results = []
    bleu_scores = []
    meteor_scores = []
    char_match_scores = []
    
    logger.info(f"Starting evaluation of {len(df)} samples...")
    
    for i, row in df.iterrows():
        logger.info(f"Evaluating sample {i+1}/{len(df)}")
        
        # Get source and reference texts
        source_text = row[source_col]
        reference_translation = row[target_col]
        
        # Generate model translation
        model_translation = model.translate_zh_to_th(source_text, system_prompt=system_prompt)
        
        # Compute metrics
        bleu = compute_bleu(reference_translation, model_translation)
        meteor = compute_meteor(reference_translation, model_translation)
        char_match = compute_character_match_ratio(reference_translation, model_translation)
        
        # Store results
        result = {
            "sample_id": i,
            "source": source_text,
            "reference": reference_translation,
            "model_translation": model_translation,
            "bleu": bleu,
            "meteor": meteor,
            "char_match": char_match
        }
        results.append(result)
        
        # Collect scores for summary statistics
        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        char_match_scores.append(char_match)
    
    # Compute summary statistics
    summary = {
        "avg_bleu": np.mean(bleu_scores),
        "avg_meteor": np.mean(meteor_scores),
        "avg_char_match": np.mean(char_match_scores),
        "median_bleu": np.median(bleu_scores),
        "median_meteor": np.median(meteor_scores),
        "median_char_match": np.median(char_match_scores),
        "min_bleu": np.min(bleu_scores),
        "min_meteor": np.min(meteor_scores),
        "min_char_match": np.min(char_match_scores),
        "max_bleu": np.max(bleu_scores),
        "max_meteor": np.max(meteor_scores),
        "max_char_match": np.max(char_match_scores),
    }
    
    # Display summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Number of samples: {len(df)}")
    logger.info(f"Average BLEU score: {summary['avg_bleu']:.4f}")
    logger.info(f"Average METEOR score: {summary['avg_meteor']:.4f}")
    logger.info(f"Average Character Match Ratio: {summary['avg_char_match']:.4f}")
    
    # Save results if output path is provided
    if output_path:
        logger.info(f"Saving evaluation results to {output_path}")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        
        # Also save a summary text file
        summary_path = os.path.splitext(output_path)[0] + "_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# XmodelLM1.5 Translation Evaluation Summary\n\n")
            f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
            f.write(f"Number of samples: {len(df)}\n\n")
            f.write("## Metrics\n\n")
            f.write(f"- Average BLEU score: {summary['avg_bleu']:.4f}\n")
            f.write(f"- Average METEOR score: {summary['avg_meteor']:.4f}\n")
            f.write(f"- Average Character Match Ratio: {summary['avg_char_match']:.4f}\n\n")
            f.write("## Score Distributions\n\n")
            f.write(f"- BLEU: min={summary['min_bleu']:.4f}, median={summary['median_bleu']:.4f}, max={summary['max_bleu']:.4f}\n")
            f.write(f"- METEOR: min={summary['min_meteor']:.4f}, median={summary['median_meteor']:.4f}, max={summary['max_meteor']:.4f}\n")
            f.write(f"- Character Match: min={summary['min_char_match']:.4f}, median={summary['median_char_match']:.4f}, max={summary['max_char_match']:.4f}\n")
        
        logger.info(f"Summary saved to {summary_path}")
    
    return {"results": results, "summary": summary}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate XmodelLM1.5 translations against reference dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV with source and target columns")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Path to save evaluation results")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    
    args = parser.parse_args()
    
    evaluate_translations(args.dataset, args.output, args.samples)

if __name__ == "__main__":
    main()
