"""
Generate a full-sized dataset using the improved DeepSeek Reasoner implementation.

This script generates a comprehensive medical dialogue dataset using the 
improved implementation with DeepSeek Reasoner and analyzes its diversity.
"""
import argparse
import os
import sys
import logging
import time
import pandas as pd
from datetime import datetime
import subprocess

# Add the src directory to the path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

# Import the generate_dataset and analyze_dataset modules
from generate_dataset import generate_deepseek_medical_dialogue

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_and_analyze(n_samples=100, output_dir="data", analyze=True):
    """
    Generate a dataset and analyze its diversity.
    
    Args:
        n_samples (int): Number of samples to generate
        output_dir (str): Directory to save the generated dataset
        analyze (bool): Whether to analyze the dataset after generation
        
    Returns:
        str: Path to the generated dataset
    """
    # Generate a unique file identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = ''.join(chr(ord('A') + i % 26) for i in range(5))
    filename = f"deepseek_reasoner_dataset_{file_id}_{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
      # Generate the dataset
    logger.info(f"Generating dataset with {n_samples} samples...")
    start_time = time.time()
    
    # Call the generate_deepseek_medical_dialogue function
    generate_deepseek_medical_dialogue(
        n_samples=n_samples,
        output_path=output_path,
        use_deepseek_reasoner=True,  # Ensure we're using DeepSeek Reasoner
        seed=42
    )
    
    end_time = time.time()
    generation_time = end_time - start_time
    logger.info(f"Dataset generation completed in {generation_time:.2f} seconds")
    logger.info(f"Dataset saved to {output_path}")
    
    # Analyze the dataset
    if analyze:
        logger.info("Analyzing dataset diversity...")
        try:
            # Use the analyze_dataset.py script through subprocess
            analyze_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analyze_dataset.py')
            cmd = [sys.executable, analyze_script, output_path]
            
            # Run the analysis
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Log the output
            if result.stdout:
                logger.info("Analysis results:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(line)
            
            if result.stderr:
                logger.warning("Analysis warnings/errors:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.warning(line)
                        
            logger.info(f"Analysis completed for {output_path}")
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate a full dataset using DeepSeek Reasoner")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save the dataset")
    parser.add_argument("--no-analysis", action="store_true", help="Skip analysis after generation")
    
    args = parser.parse_args()
    
    dataset_path = generate_and_analyze(
        n_samples=args.samples,
        output_dir=args.output_dir,
        analyze=not args.no_analysis
    )
    
    logger.info(f"Process completed. Dataset available at: {dataset_path}")

if __name__ == "__main__":
    main()
