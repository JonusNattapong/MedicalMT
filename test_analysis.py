# -*- coding: utf-8 -*-
"""
Script to test new dataset analysis feature
"""
import sys
import os
import pandas as pd

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import analyze_dataset function
from src.analyze_dataset import analyze_dataset

def main():
    print("===== Testing Dataset Analysis Feature =====")
    
    # Define paths
    dataset_path = "data/qa_train.csv"
    output_dir = "data/analysis_test"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print info
    print(f"Dataset: {dataset_path}")
    print(f"Output directory: {output_dir}")
    
    # Test if file exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset file {dataset_path} not found")
        return
        
    # Get file size
    file_size = os.path.getsize(dataset_path)
    print(f"File size: {file_size} bytes")
    
    # Read a few lines to verify content
    df = pd.read_csv(dataset_path)
    print(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
    print("Columns:", df.columns.tolist())
    
    # Run analysis
    print("\nStarting analysis...")
    try:
        metrics = analyze_dataset(dataset_path, output_dir)
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}")
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
    
if __name__ == "__main__":
    main()
