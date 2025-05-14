"""
RLHF (Reinforcement Learning from Human Feedback) for MedMT

This script implements RLHF to fine-tune models using human or automated feedback.
It supports both traditional RLHF and simpler preference optimization approaches.

Usage:
    python src/rlhf_training.py --model models/MedicalZbitxLM --feedback data/medical_evaluation_MRVCGK_20250513165918.json --output models/MedicalZbitxLM-rlhf
"""

import argparse
import os
import yaml
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)
from src.model import MedMTModel
from src.data_loader import load_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure deterministic behavior
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FeedbackDataset(Dataset):
    """Dataset for training with feedback"""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract data
        context = item.get('context', '')
        source = item.get('source', '')
        preferred = item.get('preferred', '')  # The preferred translation
        rejected = item.get('rejected', '')    # The rejected translation (if comparison-based)
        
        # Format input
        if context and not pd.isna(context):
            input_text = f"{context}\n{source}"
        else:
            input_text = source
        
        # Tokenize inputs
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize preferred output
        preferred_ids = self.tokenizer(
            preferred,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Tokenize rejected output if available
        if rejected:
            rejected_ids = self.tokenizer(
                rejected,
                max_length=self.max_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt"
            ).input_ids
        else:
            rejected_ids = None
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "preferred_ids": preferred_ids.squeeze(),
            "rejected_ids": rejected_ids.squeeze() if rejected_ids is not None else None
        }

def load_feedback_data(feedback_path):
    """Load feedback data for RLHF training"""
    if not os.path.exists(feedback_path):
        logger.error(f"Feedback file not found: {feedback_path}")
        return []
    
    # Determine file type
    if feedback_path.endswith('.json'):
        # JSON format
        with open(feedback_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to list format if it's not already
        if isinstance(data, dict):
            data = [data]
        
        logger.info(f"Loaded {len(data)} feedback items from JSON")
        return data
    
    elif feedback_path.endswith('.csv'):
        # CSV format
        df = pd.read_csv(feedback_path)
        
        # Convert to list of dicts
        data = df.to_dict('records')
        logger.info(f"Loaded {len(data)} feedback items from CSV")
        return data
    
    else:
        logger.error(f"Unsupported file format: {feedback_path}")
        return []

def prepare_preference_data(feedback_data, test_df=None):
    """Prepare data for preference learning"""
    preference_data = []
    
    for item in feedback_data:
        # Check if this is a complete preference item
        if ('source' in item or 'context' in item) and 'preferred' in item:
            preference_data.append(item)
            continue
        
        # If it's a quality assessment but no preference
        # We'll need test_df to map to sources
        if test_df is not None and ('id' in item or 'index' in item):
            idx = item.get('id', item.get('index'))
            if idx < len(test_df):
                row = test_df.iloc[idx]
                new_item = {
                    'source': row.get('source', ''),
                    'context': row.get('context', ''),
                    'preferred': item.get('preferred_translation', item.get('best_translation', item.get('target', '')))
                }
                preference_data.append(new_item)
    
    logger.info(f"Prepared {len(preference_data)} items for preference learning")
    return preference_data

def simple_preference_optimization(model, tokenizer, feedback_data, output_dir, 
                                 batch_size=4, epochs=3, learning_rate=5e-6):
    """Simple preference optimization approach (simpler than full RLHF)"""
    logger.info("Starting simple preference optimization...")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create dataset
    dataset = FeedbackDataset(feedback_data, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    total_loss = 0
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # Move batch to device
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["preferred_ids"]
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}")
        total_loss += avg_epoch_loss
    
    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Simple preference optimization completed. Model saved to {output_dir}")
    
    return {
        "model_path": output_dir,
        "avg_loss": total_loss / epochs
    }

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer for RLHF"""
    try:
        # Try using MedMTModel
        model = MedMTModel(model_path)
        tokenizer = model.tokenizer
        base_model = model.model
        
        logger.info(f"Loaded model using MedMTModel from {model_path}")
        return tokenizer, base_model
    except Exception as e:
        logger.warning(f"Failed to load model using MedMTModel: {e}")
        logger.info("Trying to load model directly with transformers...")
        
        # Direct loading with transformers
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info(f"Loaded model directly with transformers from {model_path}")
        return tokenizer, base_model

def rlhf_compare_outputs(model, tokenizer, test_df, output_file, n_samples=10):
    """Generate and compare outputs before and after RLHF"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Randomly select samples for comparison
    if len(test_df) > n_samples:
        sample_df = test_df.sample(n_samples)
    else:
        sample_df = test_df
    
    results = []
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Comparing outputs"):
        context = row.get('context', '')
        source = row.get('source', '')
        
        # Format input
        if context and not pd.isna(context):
            input_text = f"{context}\n{source}"
        else:
            input_text = source
        
        # Generate with different parameters to show variation
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Generate with standard parameters
        with torch.no_grad():
            standard_output = model.generate(
                input_ids,
                max_length=256,
                num_beams=4,
                temperature=0.7,
            )
            standard_text = tokenizer.decode(standard_output[0], skip_special_tokens=True)
        
        # Generate with higher diversity
        with torch.no_grad():
            diverse_output = model.generate(
                input_ids,
                max_length=256,
                num_beams=4,
                temperature=0.9,
                top_p=0.92,
            )
            diverse_text = tokenizer.decode(diverse_output[0], skip_special_tokens=True)
        
        results.append({
            'context': context,
            'source': source,
            'standard_output': standard_text,
            'diverse_output': diverse_text
        })
    
    # Save results
    output_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    logger.info(f"Comparison results saved to {output_file}")
    return output_df

def main():
    parser = argparse.ArgumentParser(description="RLHF Training for MedMT")
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to base model directory or model name')
    parser.add_argument('--feedback', type=str, required=True, 
                        help='Path to feedback data file (JSON or CSV)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output directory for RLHF-tuned model')
    parser.add_argument('--test_data', type=str, default=None, 
                        help='Path to test data file (CSV) for evaluation')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-6, 
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load config if specified
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Use config values as defaults if not specified
    if not args.output:
        base_name = os.path.basename(args.model)
        args.output = os.path.join(config.get('output_dir', 'outputs'), f"{base_name}-rlhf")
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model)
    
    # Load feedback data
    feedback_data = load_feedback_data(args.feedback)
    
    if not feedback_data:
        logger.error("No feedback data found. Aborting RLHF training.")
        return
    
    # Load test data if specified
    test_df = None
    if args.test_data:
        test_df = load_data(args.test_data, is_train=False)
    
    # Prepare preference data
    preference_data = prepare_preference_data(feedback_data, test_df)
    
    if not preference_data:
        logger.error("Failed to prepare preference data. Aborting RLHF training.")
        return
    
    # Run simple preference optimization
    result = simple_preference_optimization(
        model=model,
        tokenizer=tokenizer,
        feedback_data=preference_data,
        output_dir=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Compare outputs if test data is available
    if test_df is not None:
        comparison_file = os.path.join(os.path.dirname(args.output), "rlhf_comparison.csv")
        rlhf_compare_outputs(model, tokenizer, test_df, comparison_file)

if __name__ == "__main__":
    main()
