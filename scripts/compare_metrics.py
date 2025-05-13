"""
Compare and visualize metrics from different model translations.

This script loads evaluation results from multiple models and generates
comparison visualizations to help understand each model's strengths.
"""
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_evaluation_results(results_files):
    """
    Load evaluation results from multiple files.
    
    Args:
        results_files (dict): Dictionary with model names as keys and result file paths as values
        
    Returns:
        dict: Dictionary with model names as keys and loaded dataframes as values
    """
    results = {}
    
    for model_name, file_path in results_files.items():
        try:
            logger.info(f"Loading results for {model_name} from {file_path}")
            df = pd.read_csv(file_path)
            results[model_name] = df
            logger.info(f"Loaded {len(df)} samples for {model_name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            
    return results

def extract_metrics(results):
    """
    Extract metrics from evaluation results for comparison.
    
    Args:
        results (dict): Dictionary with model names as keys and dataframes as values
        
    Returns:
        pd.DataFrame: DataFrame with metrics for each model
    """
    metrics_data = []
    
    for model_name, df in results.items():
        # Extract metrics columns
        metric_cols = [col for col in df.columns if col in ['bleu', 'meteor', 'char_match']]
        
        if not metric_cols:
            logger.warning(f"No metric columns found in results for {model_name}")
            continue
            
        # Compute average metrics
        metrics = {col: df[col].mean() for col in metric_cols}
        metrics['model'] = model_name
        metrics_data.append(metrics)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    return metrics_df

def visualize_metrics(metrics_df, output_dir):
    """
    Create visualizations for model comparison.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with metrics for each model
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create bar plot for all metrics
    ax = sns.barplot(data=pd.melt(metrics_df, id_vars=['model'], 
                               value_vars=[c for c in metrics_df.columns if c != 'model'],
                               var_name='Metric', value_name='Score'),
                  x='model', y='Score', hue='Metric')
    
    plt.title('Translation Quality Metrics Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Metric', title_fontsize=12)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'metrics_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved visualization to {output_path}")
    
    # Create individual metric plots
    for metric in [c for c in metrics_df.columns if c != 'model']:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='model', y=metric)
        plt.title(f'{metric.upper()} Score Comparison', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel(f'{metric.upper()} Score', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        metric_output_path = os.path.join(output_dir, f'{metric}_comparison_{timestamp}.png')
        plt.savefig(metric_output_path, dpi=300)
        logger.info(f"Saved {metric} visualization to {metric_output_path}")
    
    # Create summary table
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    pd_table = pd.DataFrame(metrics_df).set_index('model')
    table = plt.table(cellText=pd_table.values.round(4),
                     rowLabels=pd_table.index,
                     colLabels=pd_table.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title('Translation Metrics Summary', fontsize=16)
    plt.tight_layout()
    
    # Save table
    table_output_path = os.path.join(output_dir, f'metrics_table_{timestamp}.png')
    plt.savefig(table_output_path, dpi=300)
    logger.info(f"Saved metrics table to {table_output_path}")
    
    # Create a CSV with the summary
    csv_output_path = os.path.join(output_dir, f'metrics_summary_{timestamp}.csv')
    metrics_df.to_csv(csv_output_path, index=False)
    logger.info(f"Saved metrics summary to {csv_output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Compare metrics from different translation models")
    parser.add_argument("--results", type=str, required=True, nargs='+',
                      help="List of evaluation result CSV files in format model_name:file_path")
    parser.add_argument("--output-dir", type=str, default="results",
                      help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Parse results files format
    results_files = {}
    for result_arg in args.results:
        try:
            model_name, file_path = result_arg.split(':', 1)
            results_files[model_name] = file_path
        except ValueError:
            logger.error(f"Invalid format for --results argument: {result_arg}. Should be model_name:file_path")
            sys.exit(1)
    
    # Load results
    results = load_evaluation_results(results_files)
    
    if not results:
        logger.error("No valid results loaded. Exiting.")
        sys.exit(1)
    
    # Extract metrics
    metrics_df = extract_metrics(results)
    
    # Visualize metrics
    visualize_metrics(metrics_df, args.output_dir)
    
    logger.info("Comparison completed successfully")

if __name__ == "__main__":
    main()
