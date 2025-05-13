# -*- coding: utf-8 -*-
"""
Translation quality assessment utilities for MedMT
"""
import re
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Dict, List, Any, Tuple
from .display_utils import ProgressBar
from .response_cache import cached_api_call
import logging

def calculate_simple_metrics(source: str, translation: str) -> Dict[str, float]:
    """Calculate simple metrics for translation quality assessment"""
    metrics = {}
    
    # Length ratio (Thai translation should be somewhat proportional to Chinese source)
    source_length = len(source)
    translation_length = len(translation)
    metrics["length_ratio"] = translation_length / max(source_length, 1)
    
    # Thai character ratio (measure of how much is actually Thai)
    thai_char_count = len([c for c in translation if '\u0E00' <= c <= '\u0E7F'])
    metrics["thai_char_ratio"] = thai_char_count / max(translation_length, 1)
    
    # Question mark preservation
    has_question_zh = "吗" in source or "?" in source
    has_question_th = "?" in translation or "ไหม" in translation or "หรือไม่" in translation
    metrics["question_preserved"] = 1.0 if has_question_zh == has_question_th else 0.0
    
    return metrics

def assess_translations_with_model(df: pd.DataFrame, sample_ratio: float = 0.1, min_samples: int = 5) -> pd.DataFrame:
    """
    Assess translations using DeepSeek model for a sample of the dataset
    
    Args:
        df: DataFrame containing translations with 'source' and 'target' columns
        sample_ratio: Proportion of dataset to assess (0-1)
        min_samples: Minimum number of samples to assess
    
    Returns:
        DataFrame with added assessment columns
    """
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[WARNING] DEEPSEEK_API_KEY not found, skipping model-based assessment")
        return df
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    # Determine sample size
    total_rows = len(df)
    sample_size = max(min_samples, int(total_rows * sample_ratio))
    sample_size = min(sample_size, total_rows)  # Don't exceed dataset size
    
    print(f"[INFO] Assessing {sample_size} translation samples with DeepSeek model")
    
    # Select samples
    sampled_indices = np.random.choice(df.index, size=sample_size, replace=False)
    
    # Initialize columns
    df.loc[sampled_indices, "qe_score"] = np.nan
    df.loc[sampled_indices, "qe_feedback"] = ""
    
    # Assessment prompt template
    assessment_prompt = """
    请评估以下中泰医疗翻译的质量:

    中文原文: {source}
    泰文翻译: {target}

    评估标准:
    1. 完整性：翻译是否包含源文本的所有信息
    2. 准确性：医学术语和概念是否准确翻译
    3. 流畅度：泰语表达是否自然流畅
    4. 语法正确性：泰语语法是否准确
    5. 风格一致性：是否保持了医疗专业语言风格

    请按照1-10分的比例给出一个综合评分（10分为最高），并用1-2句话简短说明评分原因。
    回复格式:
    分数: [1-10]
    原因: [简短评价]
    """
    
    # Progress bar
    progress = ProgressBar(sample_size, prefix='评估翻译质量:', suffix='完成')
    
    for idx in sampled_indices:
        source = df.loc[idx, "source"]
        target = df.loc[idx, "target"]
        
        try:
            response_dict = cached_api_call(
                client,
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": assessment_prompt.format(source=source, target=target)}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            response_text = response_dict["choices"][0]["message"]["content"]
            
            # Extract score and reason
            score_match = re.search(r'分数:\s*(\d+(?:\.\d+)?)', response_text)
            reason_match = re.search(r'原因:\s*(.+)', response_text)
            
            if score_match:
                score = float(score_match.group(1))
                df.loc[idx, "qe_score"] = min(10.0, max(1.0, score))  # Ensure score is between 1-10
            
            if reason_match:
                df.loc[idx, "qe_feedback"] = reason_match.group(1).strip()
        
        except Exception as e:
            logging.error(f"Assessment error for index {idx}: {e}")
        
        progress.increment()
    
    # Calculate dataset-level statistics
    assessed_df = df[df["qe_score"].notna()]
    if not assessed_df.empty:
        avg_score = assessed_df["qe_score"].mean()
        print(f"[INFO] Assessment complete. Average translation quality score: {avg_score:.2f}/10")
    
    return df

def run_quality_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a comprehensive quality assessment on the generated dataset
    
    Args:
        df: DataFrame containing translations with 'source' and 'target' columns
    
    Returns:
        DataFrame with added assessment metrics
    """
    print("[INFO] Running translation quality assessment...")
    
    # 1. Calculate simple metrics for all translations
    metrics = []
    for i, row in df.iterrows():
        source = row["source"] if "source" in df.columns else row.get("question_zh", "")
        target = row["target"] if "target" in df.columns else row.get("question_th", "")
        
        if source and target:
            row_metrics = calculate_simple_metrics(source, target)
            metrics.append(row_metrics)
        else:
            metrics.append({
                "length_ratio": np.nan,
                "thai_char_ratio": np.nan,
                "question_preserved": np.nan
            })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Add metrics to original dataframe
    for col in metrics_df.columns:
        df[f"metric_{col}"] = metrics_df[col]
    
    # 2. Model-based assessment for a sample
    df = assess_translations_with_model(df, sample_ratio=0.05, min_samples=10)
    
    # 3. Generate summary statistics
    valid_metrics = df[df["metric_thai_char_ratio"] > 0.5]  # Filter out failed translations
    
    summary = {
        "total_samples": len(df),
        "valid_samples": len(valid_metrics),
        "avg_length_ratio": valid_metrics["metric_length_ratio"].mean(),
        "avg_thai_char_ratio": valid_metrics["metric_thai_char_ratio"].mean(),
        "question_preservation_rate": valid_metrics["metric_question_preserved"].mean(),
    }
    
    if "qe_score" in df.columns and df["qe_score"].notna().any():
        summary["avg_quality_score"] = df["qe_score"].mean()
    
    print("\n[Translation Quality Assessment Summary]")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Valid translations: {summary['valid_samples']} ({summary['valid_samples']/max(summary['total_samples'], 1)*100:.1f}%)")
    print(f"Average length ratio (Thai/Chinese): {summary['avg_length_ratio']:.2f}")
    print(f"Average Thai character ratio: {summary['avg_thai_char_ratio']:.2f}")
    print(f"Question preservation rate: {summary['question_preservation_rate']:.2f}")
    
    if "avg_quality_score" in summary:
        print(f"Average quality score (1-10): {summary['avg_quality_score']:.2f}")
    
    return df
