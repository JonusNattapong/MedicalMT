# -*- coding: utf-8 -*-
"""
File utilities for MedMT dataset generation
"""
import os
import random
import string
import datetime
import logging
import sys
import pandas as pd

def random_id(prefix="D"):
    """Generate a random ID for dataset files"""
    chars = string.ascii_uppercase + string.digits
    return f"{prefix}{''.join(random.choices(chars, k=5))}"

def get_datasets_dir(base_dir="data"):
    """Get or create datasets directory"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def unique_dataset_filename(base_filename, extension, prefix="D"):
    """Generate a unique filename for the dataset"""
    base, ext = os.path.splitext(base_filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = random_id(prefix)
    return f"{base}_{unique_id}_{timestamp}{extension}"

def save_dataset(df, output_path_suggestion, file_format):
    """Saves the DataFrame to a uniquely named file in the specified or default directory."""
    # Extract directory and filename
    output_dir = os.path.dirname(output_path_suggestion) if output_path_suggestion else get_datasets_dir()
    base_name = (os.path.basename(output_path_suggestion) if output_path_suggestion 
                else "dataset.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get actual file extension based on format
    ext_map = {
        'csv': '.csv',
        'json': '.json',
        'jsonl': '.jsonl',
        'txt': '.txt',
        'arrow': '.arrow',
        'parquet': '.parquet'
    }
    actual_ext = ext_map.get(file_format, '.csv')
    
    # Generate unique filename
    prefix_char = base_name[0].upper() if base_name else 'D'
    filename = unique_dataset_filename(base_name, actual_ext, prefix=prefix_char)
    output_filepath = os.path.join(output_dir, filename)

    try:
        if file_format == 'csv':
            df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        elif file_format == 'json':
            df.to_json(output_filepath, orient='records', force_ascii=False, indent=2)
        elif file_format == 'jsonl':
            df.to_json(output_filepath, orient='records', force_ascii=False, lines=True)
        elif file_format == 'txt':
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for record in df.to_dict(orient='records'):
                    f.write(str(record) + '\n')
        elif file_format == 'arrow' or file_format == 'parquet':
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(df)
                if file_format == 'arrow':
                    with pa.OSFile(output_filepath, 'wb') as sink:
                        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                            writer.write_table(table)
                else:  # parquet
                    pq.write_table(table, output_filepath)
            except ImportError:
                print(f"[ERROR] pyarrow is not installed. Cannot save as {file_format.upper()}. Please install it using 'pip install pyarrow'.")
                print(f"[INFO] Skipping save for {output_filepath}.")
                return

        return output_filepath, len(df)
    except Exception as e:
        print(f"[ERROR] Failed to save dataset to {output_filepath}. Error: {e}")
        logging.error(f"FileSaveError | Sample: 0 | Failed to save {output_filepath}: {e}")

def recommend_format(mode, output_filename_suggestion, n_samples):
    """Recommend the best file format based on mode, output filename, and sample size."""
    # Check if user already specified a valid extension in the output filename
    if output_filename_suggestion:
        _, ext = os.path.splitext(output_filename_suggestion)
        ext = ext.lower().lstrip('.')
        if ext in ['csv', 'json', 'jsonl', 'txt', 'arrow', 'parquet']:
            return ext
            
    # General recommendations
    if n_samples > 5_000_000:  # For very large datasets, binary formats are better
        return "parquet" if 'pyarrow' in sys.modules else "arrow"  # Parquet is often more compressed
    if n_samples > 1_000_000:
        return "arrow" if 'pyarrow' in sys.modules else "csv"

    # Mode-specific recommendations for smaller datasets
    if mode in ["dialogue", "qa", "reasoning", "summarization", "classification"]:
        return "csv"  # CSV is widely compatible and good for tabular data
    elif mode == "mrc":  # Machine Reading Comprehension often uses JSON (like SQuAD)
        return "json"
    elif mode == "pretrain":  # Pretraining data often uses plain text or JSONL
        return "jsonl" if n_samples > 10000 else "txt"
    
    return "csv"  # Default fallback