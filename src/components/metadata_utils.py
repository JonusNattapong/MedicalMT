# -*- coding: utf-8 -*-
"""
Metadata utilities for MedMT dataset generation
"""
import pandas as pd

def add_metadata(df):
    """Add metadata to the dataset"""
    metadata = {
        "generator": "DeepSeek-Ai",
        "model": "deepseek-chat",
        "license": "CC BY-SA-NC 4.0",
        "generation_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "version": "2.0.0", # Corresponds to the version of this script/dataset structure
    }
    
    # Add metadata as additional columns
    for key, value in metadata.items():
        df[f"_metadata_{key}"] = value
    
    return df