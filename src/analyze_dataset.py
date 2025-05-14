# -*- coding: utf-8 -*-
"""
Analyze MedMT dataset to evaluate diversity, quality, and other metrics
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
import os
from wordcloud import WordCloud
import re
from typing import Dict, List, Any
import seaborn as sns
import sys
from pathlib import Path

def extract_keywords(text, min_length=1):
    """Extract keywords from text"""
    # Simple regex to extract words
    if not text or not isinstance(text, str):
        return []
        
    # Extract Thai words
    if re.search(r'[\u0E00-\u0E7F]', text):  # Contains Thai characters
        words = re.findall(r'[\u0E00-\u0E7F]+', text)
    else:  # Assume Chinese
        words = list(text)  # Character by character for Chinese
        
    return [w for w in words if len(w) >= min_length]

def calculate_diversity_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate diversity metrics for the dataset"""
    metrics = {}
    
    # Basic stats
    metrics['total_samples'] = len(df)
    
    # Source text diversity
    if 'source' in df.columns:
        source_texts = df['source'].tolist()
        source_words = []
        for text in source_texts:
            source_words.extend(extract_keywords(text))
            
        source_word_counts = Counter(source_words)
        
        metrics['unique_source_words'] = len(source_word_counts)
        metrics['source_vocabulary_richness'] = len(source_word_counts) / max(1, sum(source_word_counts.values()))
        
        # Most common words
        most_common = source_word_counts.most_common(10)
        metrics['most_common_source_words'] = [
            {'word': word, 'count': count} for word, count in most_common
        ]
        
    # Target text diversity
    if 'target' in df.columns:
        target_texts = df['target'].tolist()
        target_words = []
        for text in target_texts:
            target_words.extend(extract_keywords(text))
            
        target_word_counts = Counter(target_words)
        
        metrics['unique_target_words'] = len(target_word_counts)
        metrics['target_vocabulary_richness'] = len(target_word_counts) / max(1, sum(target_word_counts.values()))
        
        # Most common words
        most_common = target_word_counts.most_common(10)
        metrics['most_common_target_words'] = [
            {'word': word, 'count': count} for word, count in most_common
        ]
    
    # Text length analysis
    if 'source' in df.columns and 'target' in df.columns:
        df['source_length'] = df['source'].apply(lambda x: len(str(x)))
        df['target_length'] = df['target'].apply(lambda x: len(str(x)))
        
        metrics['avg_source_length'] = df['source_length'].mean()
        metrics['avg_target_length'] = df['target_length'].mean()
        metrics['avg_length_ratio'] = df['target_length'].mean() / max(1, df['source_length'].mean())
        
        # Length distribution
        metrics['source_length_percentiles'] = {
            'min': df['source_length'].min(),
            '25%': df['source_length'].quantile(0.25),
            'median': df['source_length'].median(),
            '75%': df['source_length'].quantile(0.75),
            'max': df['source_length'].max()
        }
        
        metrics['target_length_percentiles'] = {
            'min': df['target_length'].min(),
            '25%': df['target_length'].quantile(0.25),
            'median': df['target_length'].median(),
            '75%': df['target_length'].quantile(0.75),
            'max': df['target_length'].max()
        }
    
    # Quality metrics if available
    if 'qe_score' in df.columns and df['qe_score'].notna().any():
        metrics['avg_quality_score'] = df['qe_score'].mean()
        
    return metrics

def generate_visualizations(df: pd.DataFrame, output_dir: str):
    """Generate visualizations for the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Text length distribution
    if 'source' in df.columns and 'target' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Add source length histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df['source'].apply(lambda x: len(str(x))), kde=True)
        plt.title('Source Text Length Distribution')
        plt.xlabel('Length (characters)')
        plt.ylabel('Frequency')
        
        # Add target length histogram
        plt.subplot(1, 2, 2)
        sns.histplot(df['target'].apply(lambda x: len(str(x))), kde=True)
        plt.title('Target Text Length Distribution')
        plt.xlabel('Length (characters)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'length_distribution.png'))
        plt.close()
    
    # 2. Word clouds
    if 'source' in df.columns:
        try:
            text = ' '.join([str(t) for t in df['source'].tolist()])
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white', 
                                max_words=100, 
                                font_path='SimHei.ttf' if os.path.exists('SimHei.ttf') else None)
            wordcloud.generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Source Text Word Cloud')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'source_wordcloud.png'))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate source wordcloud - {e}")
    
    if 'target' in df.columns:
        try:
            text = ' '.join([str(t) for t in df['target'].tolist()])
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white', 
                                max_words=100, 
                                font_path='THSarabun.ttf' if os.path.exists('THSarabun.ttf') else None)
            wordcloud.generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Target Text Word Cloud')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'target_wordcloud.png'))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate target wordcloud - {e}")
    
    # 3. Quality metrics if available
    if 'qe_score' in df.columns and df['qe_score'].notna().any():
        plt.figure(figsize=(8, 6))
        sns.histplot(df['qe_score'].dropna(), kde=True)
        plt.title('Translation Quality Score Distribution')
        plt.xlabel('Quality Score (1-10)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_distribution.png'))
        plt.close()

def generate_report(metrics: Dict[str, Any], output_path: str):
    """Generate a markdown report with dataset analysis"""
    report = [
        "# MedMT Dataset Analysis Report",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Overview",
        f"- Total samples: {metrics['total_samples']}",
        ""
    ]
    
    # Vocabulary stats
    if 'unique_source_words' in metrics:
        report.extend([
            "## Vocabulary Statistics",
            f"- Unique words in source texts: {metrics['unique_source_words']}",
            f"- Source vocabulary richness: {metrics['source_vocabulary_richness']:.4f}",
            f"- Unique words in target texts: {metrics.get('unique_target_words', 'N/A')}",
            f"- Target vocabulary richness: {metrics.get('target_vocabulary_richness', 'N/A'):.4f}",
            "",
            "### Most Common Source Words",
        ])
        
        for item in metrics.get('most_common_source_words', []):
            report.append(f"- {item['word']}: {item['count']}")
        
        report.append("")
        report.append("### Most Common Target Words")
        
        for item in metrics.get('most_common_target_words', []):
            report.append(f"- {item['word']}: {item['count']}")
        
        report.append("")
    
    # Text length analysis
    if 'avg_source_length' in metrics:
        report.extend([
            "## Text Length Analysis",
            f"- Average source text length: {metrics['avg_source_length']:.2f} characters",
            f"- Average target text length: {metrics['avg_target_length']:.2f} characters",
            f"- Average length ratio (target/source): {metrics['avg_length_ratio']:.2f}",
            "",
            "### Source Length Distribution",
            f"- Minimum: {metrics['source_length_percentiles']['min']}",
            f"- 25th percentile: {metrics['source_length_percentiles']['25%']:.2f}",
            f"- Median: {metrics['source_length_percentiles']['median']:.2f}",
            f"- 75th percentile: {metrics['source_length_percentiles']['75%']:.2f}",
            f"- Maximum: {metrics['source_length_percentiles']['max']}",
            "",
            "### Target Length Distribution",
            f"- Minimum: {metrics['target_length_percentiles']['min']}",
            f"- 25th percentile: {metrics['target_length_percentiles']['25%']:.2f}",
            f"- Median: {metrics['target_length_percentiles']['median']:.2f}",
            f"- 75th percentile: {metrics['target_length_percentiles']['75%']:.2f}",
            f"- Maximum: {metrics['target_length_percentiles']['max']}",
            ""
        ])
    
    # Quality metrics if available
    if 'avg_quality_score' in metrics:
        report.extend([
            "## Quality Assessment",
            f"- Average quality score: {metrics['avg_quality_score']:.2f}/10",
            ""
        ])
    
    # Write report to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

def analyze_dataset(dataset_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Analyze a MedMT dataset and generate reports
    
    Args:
        dataset_path: Path to the dataset file
        output_dir: Directory for output files (default: same directory as dataset)
    
    Returns:
        Dictionary of metrics
    """
    print(f"Analyzing dataset: {dataset_path}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(dataset_path), 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    file_ext = os.path.splitext(dataset_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(dataset_path)
    elif file_ext in ['.json', '.jsonl']:
        df = pd.read_json(dataset_path, lines=file_ext == '.jsonl')
    elif file_ext == '.parquet':
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns")
    
    # Calculate metrics
    print("Calculating diversity metrics...")
    metrics = calculate_diversity_metrics(df)
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(df, output_dir)
    
    # Generate report
    report_path = os.path.join(output_dir, 'analysis_report.md')
    print(f"Generating analysis report: {report_path}")
    generate_report(metrics, report_path)
    
    # Convert numpy types in metrics to Python native types before saving to JSON
    metrics_converted = convert_numpy_types(metrics)
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_converted, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Analyze MedMT dataset")
    parser.add_argument("dataset_path", type=str, help="Path to dataset file")
    parser.add_argument("--output-dir", type=str, default=None, 
                      help="Directory for output files (default: ./analysis)")
    
    args = parser.parse_args()
    
    try:
        metrics = analyze_dataset(args.dataset_path, args.output_dir)
        
        # Print summary to console
        print("\n=== Dataset Analysis Summary ===")
        print(f"Total samples: {metrics['total_samples']}")
        
        if 'unique_source_words' in metrics:
            print(f"Unique words (source): {metrics['unique_source_words']}")
            print(f"Unique words (target): {metrics.get('unique_target_words', 'N/A')}")
        
        if 'avg_source_length' in metrics:
            print(f"Avg. text length: {metrics['avg_source_length']:.2f} (source), " + 
                  f"{metrics['avg_target_length']:.2f} (target)")
        
        if 'avg_quality_score' in metrics:
            print(f"Avg. quality score: {metrics['avg_quality_score']:.2f}/10")
            
        return 0
    
    except Exception as e:
        print(f"Error analyzing dataset: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
