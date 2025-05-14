# -*- coding: utf-8 -*-
"""
Data validation toolkit for Medical MT datasets

This script provides tools to validate and verify the quality of 
Chinese to Thai medical translation datasets.

Features:
1. Detect correct language in source/target pairs
2. Validate completeness of translations
3. Check for common translation errors
4. Detect formatting issues
5. Report quality metrics

Usage:
    python src/validate_dataset.py --input data/high_quality/mixed_dataset.csv
                                 --output data/validated
                                 --fix True
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Column mappings for different dataset types
COLUMN_MAPPINGS = {
    "translation": {
        "source_col": "source", 
        "target_col": "target"
    },
    "qa": {
        "source_col": "question_zh", 
        "target_col": "answer_th"
    },
    "summarization": {
        "source_col": "summary_zh", 
        "target_col": "summary_th"
    },
    "reasoning": {
        "source_col": "question_zh", 
        "target_col": "answer_th"
    }
}

# Patterns for common errors
ERROR_PATTERNS = {
    "untranslated_chinese": r"[\u4e00-\u9fff]",                    # Chinese characters in Thai text
    "html_tags": r"<[^>]+>",                                       # HTML tags
    "control_chars": r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]",     # Control characters
    "script_tags": r"<script.*?>.*?</script>",                     # Javascript
    "multiple_spaces": r"\s{2,}",                                  # Multiple spaces
    "long_repetition": r"(.{5,})\1{3,}"                            # Long text repeated many times
}

def detect_language(text: str) -> Optional[str]:
    """
    Detect if text is primarily Chinese, Thai, or another language
    
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

def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Detect dataset type from its columns
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dataset type ('translation', 'qa', 'summarization', 'reasoning', or 'unknown')
    """
    columns = set(df.columns)
    
    if {'source', 'target'}.issubset(columns):
        return 'translation'
    elif {'question_zh', 'answer_th'}.issubset(columns):
        return 'qa' if 'context' in columns else 'reasoning'  
    elif {'summary_zh', 'summary_th'}.issubset(columns):
        return 'summarization'
    else:
        return 'unknown'

def validate_parallel_text(row: pd.Series, source_col: str, target_col: str) -> Dict[str, Any]:
    """
    Validate a parallel text pair
    
    Args:
        row: DataFrame row containing source and target text
        source_col: Name of the source column
        target_col: Name of the target column
        
    Returns:
        Dictionary of validation results
    """
    source = row[source_col]
    target = row[target_col]
    
    results = {
        'valid': True,
        'errors': [],
        'source_lang': detect_language(source),
        'target_lang': detect_language(target),
        'source_len': len(source) if isinstance(source, str) else 0,
        'target_len': len(target) if isinstance(target, str) else 0,
        'error_details': {}
    }
    
    # Check for empty or missing texts
    if not source or not isinstance(source, str) or not source.strip():
        results['valid'] = False
        results['errors'].append('empty_source')
    
    if not target or not isinstance(target, str) or not target.strip():
        results['valid'] = False
        results['errors'].append('empty_target')
    
    # If either is empty, return early
    if 'empty_source' in results['errors'] or 'empty_target' in results['errors']:
        return results
    
    # Check language match
    if results['source_lang'] != 'zh':
        results['valid'] = False
        results['errors'].append('source_not_chinese')
    
    if results['target_lang'] != 'th':
        results['valid'] = False
        results['errors'].append('target_not_thai')
    
    # Check length ratio
    if results['source_len'] > 0 and results['target_len'] > 0:
        ratio = results['source_len'] / results['target_len']
        if ratio < 0.2:
            results['valid'] = False
            results['errors'].append('target_too_long')
        elif ratio > 5.0:
            results['valid'] = False
            results['errors'].append('target_too_short')
    
    # Check for common errors in target text
    for error_name, pattern in ERROR_PATTERNS.items():
        if re.search(pattern, target):
            matches = re.findall(pattern, target)
            results['valid'] = False
            results['errors'].append(error_name)
            results['error_details'][error_name] = matches[:3]  # Store up to 3 examples
    
    return results

def fix_translation_errors(text: str, errors: List[str], error_details: Dict[str, Any]) -> str:
    """
    Attempt to fix common translation errors
    
    Args:
        text: The text to fix
        errors: List of error types found
        error_details: Details of the errors
        
    Returns:
        Fixed text (or original if cannot be fixed)
    """
    if not text or not isinstance(text, str):
        return text
    
    fixed_text = text
    
    if 'multiple_spaces' in errors:
        fixed_text = re.sub(r'\s{2,}', ' ', fixed_text).strip()
    
    if 'control_chars' in errors:
        fixed_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', fixed_text)
    
    if 'html_tags' in errors:
        fixed_text = re.sub(r'<[^>]+>', '', fixed_text)
    
    if 'script_tags' in errors:
        fixed_text = re.sub(r'<script.*?>.*?</script>', '', fixed_text, flags=re.DOTALL)
    
    # Untranslated Chinese in Thai text is complex to fix automatically
    # But we can attempt to remove isolated Chinese characters
    if 'untranslated_chinese' in errors:
        # Only remove Chinese if it's less than 10% of the text
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', fixed_text)
        if len(chinese_chars) / len(fixed_text) < 0.1:
            # Remove isolated Chinese characters
            fixed_text = re.sub(r'[\u4e00-\u9fff]{1,2}', '', fixed_text)
            # Clean up any resulting double spaces
            fixed_text = re.sub(r'\s{2,}', ' ', fixed_text).strip()
    
    return fixed_text

def validate_dataset(df: pd.DataFrame, fix_errors: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate an entire dataset
    
    Args:
        df: DataFrame to validate
        fix_errors: Whether to attempt to fix errors
        
    Returns:
        Tuple of (fixed DataFrame, validation report)
    """
    if df.empty:
        return df, {'error': 'Empty dataset'}
    
    # Detect dataset type
    dataset_type = detect_dataset_type(df)
    logger.info(f"Detected dataset type: {dataset_type}")
    
    if dataset_type == 'unknown':
        logger.warning("Unknown dataset structure, using first two columns as source/target")
        source_col = df.columns[0]
        target_col = df.columns[1]
    else:
        source_col = COLUMN_MAPPINGS[dataset_type]['source_col']
        target_col = COLUMN_MAPPINGS[dataset_type]['target_col']
    
    logger.info(f"Using columns: source={source_col}, target={target_col}")
    
    # Check if columns exist
    if source_col not in df.columns or target_col not in df.columns:
        return df, {'error': f'Missing required columns: {source_col}, {target_col}'}
    
    # Validate each row
    results = []
    fixed_df = df.copy()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating dataset"):
        validation = validate_parallel_text(row, source_col, target_col)
        validation['index'] = idx
        results.append(validation)
        
        # Fix errors if requested
        if fix_errors and not validation['valid'] and 'empty_target' not in validation['errors']:
            fixed_text = fix_translation_errors(
                row[target_col], 
                validation['errors'],
                validation['error_details']
            )
            fixed_df.at[idx, target_col] = fixed_text
    
    # Compile validation report
    valid_count = sum(1 for r in results if r['valid'])
    error_counts = {}
    
    for r in results:
        for error in r['errors']:
            error_counts[error] = error_counts.get(error, 0) + 1
    
    report = {
        'dataset_type': dataset_type,
        'source_column': source_col,
        'target_column': target_col,
        'total_samples': len(df),
        'valid_samples': valid_count,
        'invalid_samples': len(df) - valid_count,
        'validity_percentage': (valid_count / len(df)) * 100 if len(df) > 0 else 0,
        'error_counts': error_counts,
        'error_percentages': {k: (v / len(df)) * 100 for k, v in error_counts.items()},
        'length_statistics': {
            'source_mean': df[source_col].str.len().mean(),
            'source_median': df[source_col].str.len().median(),
            'source_std': df[source_col].str.len().std(),
            'target_mean': df[target_col].str.len().mean(),
            'target_median': df[target_col].str.len().median(),
            'target_std': df[target_col].str.len().std(),
        }
    }
    
    return fixed_df, report

def generate_validation_report(report: Dict[str, Any], output_dir: str) -> None:
    """
    Generate a validation report with visualizations
    
    Args:
        report: Validation report dictionary
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON report
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Generate error distribution visualization
    if report.get('error_counts'):
        plt.figure(figsize=(10, 6))
        
        errors = list(report['error_counts'].keys())
        counts = list(report['error_counts'].values())
        
        # Sort by count in descending order
        sorted_indices = np.argsort(counts)[::-1]
        errors = [errors[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        plt.bar(errors, counts)
        plt.title('Error Distribution')
        plt.xlabel('Error Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
        plt.close()
    
    # Generate length distribution visualization
    if 'length_statistics' in report:
        stats = report['length_statistics']
        
        if all(k in stats for k in ['source_mean', 'target_mean']):
            plt.figure(figsize=(12, 6))
            
            # Create synthetic distributions based on mean and std
            source_mean = stats['source_mean']
            source_std = stats['source_std']
            target_mean = stats['target_mean']
            target_std = stats['target_std']
            
            x_source = np.linspace(max(0, source_mean - 3*source_std), source_mean + 3*source_std, 100)
            y_source = np.exp(-0.5 * ((x_source - source_mean) / source_std) ** 2)
            
            x_target = np.linspace(max(0, target_mean - 3*target_std), target_mean + 3*target_std, 100)
            y_target = np.exp(-0.5 * ((x_target - target_mean) / target_std) ** 2)
            
            plt.plot(x_source, y_source, label=f'Source (mean={source_mean:.1f})')
            plt.plot(x_target, y_target, label=f'Target (mean={target_mean:.1f})')
            
            plt.title('Length Distribution')
            plt.xlabel('Text Length (characters)')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, 'length_distribution.png'))
            plt.close()
    
    # Generate markdown report
    md_report_path = os.path.join(output_dir, 'validation_report.md')
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write('# Dataset Validation Report\n\n')
        
        f.write('## Overview\n\n')
        f.write(f"- **Dataset Type**: {report.get('dataset_type', 'Unknown')}\n")
        f.write(f"- **Source Column**: {report.get('source_column', 'N/A')}\n")
        f.write(f"- **Target Column**: {report.get('target_column', 'N/A')}\n")
        f.write(f"- **Total Samples**: {report.get('total_samples', 0)}\n")
        f.write(f"- **Valid Samples**: {report.get('valid_samples', 0)} ({report.get('validity_percentage', 0):.1f}%)\n")
        f.write(f"- **Invalid Samples**: {report.get('invalid_samples', 0)}\n\n")
        
        f.write('## Error Analysis\n\n')
        f.write('| Error Type | Count | Percentage |\n')
        f.write('|------------|-------|------------|\n')
        
        if report.get('error_counts'):
            for error, count in sorted(report['error_counts'].items(), key=lambda x: x[1], reverse=True):
                percentage = report['error_percentages'][error]
                f.write(f'| {error} | {count} | {percentage:.1f}% |\n')
        else:
            f.write('| No errors found | 0 | 0% |\n')
        
        f.write('\n## Length Statistics\n\n')
        f.write('| Metric | Source | Target |\n')
        f.write('|--------|--------|--------|\n')
        
        if 'length_statistics' in report:
            stats = report['length_statistics']
            f.write(f"| Mean | {stats.get('source_mean', 'N/A'):.1f} | {stats.get('target_mean', 'N/A'):.1f} |\n")
            f.write(f"| Median | {stats.get('source_median', 'N/A'):.1f} | {stats.get('target_median', 'N/A'):.1f} |\n")
            f.write(f"| Standard Deviation | {stats.get('source_std', 'N/A'):.1f} | {stats.get('target_std', 'N/A'):.1f} |\n")
        
        f.write('\n## Visualizations\n\n')
        f.write('### Error Distribution\n\n')
        f.write('![Error Distribution](error_distribution.png)\n\n')
        
        f.write('### Length Distribution\n\n')
        f.write('![Length Distribution](length_distribution.png)\n\n')
        
        f.write('\n\n*Generated automatically by validate_dataset.py*')
    
    logger.info(f"Validation report saved to {md_report_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Medical MT Dataset Validator")
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Input dataset file to validate"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/validated",
        help="Output directory for validated dataset and reports"
    )
    
    parser.add_argument(
        "--fix", 
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Attempt to fix common errors (True/False)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Load dataset
    try:
        _, ext = os.path.splitext(args.input)
        ext = ext.lower()
        
        if ext == '.csv':
            df = pd.read_csv(args.input)
        elif ext == '.json':
            df = pd.read_json(args.input)
        elif ext == '.jsonl':
            df = pd.read_json(args.input, lines=True)
        elif ext == '.parquet':
            df = pd.read_parquet(args.input)
        else:
            logger.error(f"Unsupported file format: {ext}")
            sys.exit(1)
            
        logger.info(f"Loaded {len(df)} samples from {args.input}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)
    
    # Validate dataset
    logger.info(f"Validating dataset with fix_errors={args.fix}...")
    fixed_df, report = validate_dataset(df, fix_errors=args.fix)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save fixed dataset if requested
    if args.fix:
        output_file = os.path.join(args.output, f"fixed_{os.path.basename(args.input)}")
        
        _, ext = os.path.splitext(output_file)
        ext = ext.lower()
        
        if ext == '.csv':
            fixed_df.to_csv(output_file, index=False)
        elif ext == '.json':
            fixed_df.to_json(output_file, orient='records', force_ascii=False, indent=2)
        elif ext == '.jsonl':
            fixed_df.to_json(output_file, orient='records', force_ascii=False, lines=True)
        elif ext == '.parquet':
            fixed_df.to_parquet(output_file, index=False)
        
        logger.info(f"Fixed dataset saved to {output_file}")
    
    # Generate validation report
    generate_validation_report(report, args.output)
    
    # Print summary
    print("\n=== Validation Summary ===")
    print(f"Total samples: {report.get('total_samples', 0)}")
    print(f"Valid samples: {report.get('valid_samples', 0)} ({report.get('validity_percentage', 0):.1f}%)")
    print(f"Invalid samples: {report.get('invalid_samples', 0)}")
    
    if report.get('error_counts'):
        print("\nTop errors:")
        for error, count in sorted(report['error_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {error}: {count} ({report['error_percentages'][error]:.1f}%)")
    
    print(f"\nDetailed report saved to {args.output}/validation_report.md")

if __name__ == "__main__":
    main()
