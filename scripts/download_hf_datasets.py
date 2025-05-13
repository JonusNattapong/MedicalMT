# -*- coding: utf-8 -*-
"""
Script to download datasets from Hugging Face Hub and save them locally.

Example Usage:
python Script/download_hf_datasets.py --datasets USERNAME/DATASET_NAME1 USERNAME/DATASET_NAME2
python Script/download_hf_datasets.py -d USERNAME/DATASET_NAME

Requires the 'datasets' library: pip install datasets
"""

import os
import logging
import argparse
from datasets import load_dataset, DownloadConfig, Dataset, DatasetDict, IterableDataset, IterableDatasetDict # Keep for now, though parts might become unused
from huggingface_hub import HfFolder, snapshot_download # Added snapshot_download
import requests # Added import
import json # Added import
from dotenv import load_dotenv # Add this import

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
# Determine the base path (assuming this script is in the 'Script' folder)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define the target base directory to save datasets
DEFAULT_DOWNLOAD_DIR = os.path.join(BASE_PATH, 'DatasetDownload')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face Hub.")
    parser.add_argument(
        "-d", "--datasets",
        nargs='+',  # Allows specifying one or more dataset IDs
        required=True,
        help="List of dataset IDs from Hugging Face Hub (e.g., 'username/dataset_name' or 'dataset_name')."
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"The base directory where datasets will be saved. Defaults to: {DEFAULT_DOWNLOAD_DIR}"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face Hub token for accessing private datasets. If not provided, uses cached login or public access."
    )
    parser.add_argument(
        "--save_token",
        action="store_true",
        help="If provided with --token, save the token for future use."
    )
    # subset argument is removed as snapshot_download works on the entire repo by default
    return parser.parse_args()

# --- Download Logic ---
def download_and_save_datasets(dataset_ids, base_download_dir, token=None):
    """
    Downloads datasets from Hugging Face Hub as raw files, preserving original structure.

    Args:
        dataset_ids (list): A list of dataset IDs to download.
        base_download_dir (str): The base directory to save datasets.
        token (str, optional): Hugging Face Hub token. Defaults to None.
    """
    logging.info(f"Base download directory set to: {base_download_dir}")
    if not os.path.exists(base_download_dir):
        try:
            os.makedirs(base_download_dir)
            logging.info(f"Created base directory: {base_download_dir}")
        except OSError as e:
            logging.error(f"Error creating base directory {base_download_dir}: {e}")
            return

    for dataset_id in dataset_ids:
        logging.info(f"--- Processing dataset: {dataset_id} ---")
        safe_subdir_name = dataset_id.replace('/', '_')
        target_dir = os.path.join(base_download_dir, safe_subdir_name)

        logging.info(f"Target save directory for raw files: {target_dir}")

        # The check for existing directory and skipping is removed to always attempt download.
        # snapshot_download with force_download=True will handle re-downloading.

        try:
            logging.info(f"Attempting to download raw files for '{dataset_id}' to {target_dir}...")
            snapshot_download(
                repo_id=dataset_id,
                local_dir=target_dir,
                token=token,
                repo_type="dataset",
                resume_download=True  # Ensures download resumes if interrupted and skips if complete
                # Removed all other parameters that might not be supported in this version
            )
            logging.info(f"Successfully downloaded/verified raw files for '{dataset_id}' to {target_dir}")

        except Exception as e:
            logging.error(f"Failed to download dataset '{dataset_id}' using snapshot_download: {e}")
            # Optionally, you might want to clean up target_dir if an error occurs
            # For now, it will leave any partially downloaded files.
            logging.warning(f"An error occurred during download of {dataset_id}. The directory {target_dir} might contain incomplete data.")
        logging.info(f"--- Finished processing dataset: {dataset_id} ---")

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    # Handle token saving
    if args.token and args.save_token:
        HfFolder.save_token(args.token)
        logging.info("Hugging Face token saved.")
        # Use the provided token for this run even if saving
        token_to_use = args.token
    elif args.token:
        # Use the provided token without saving it globally
        token_to_use = args.token
    else:
        # Rely on cached token or public access
        token_to_use = None

    download_and_save_datasets(
        dataset_ids=args.datasets,
        base_download_dir=args.download_dir,
        token=token_to_use
    )
    logging.info("Dataset download process finished.")
