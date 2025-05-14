"""
LoRA Configuration Generator for MedMT

This script generates LoRA configurations for parameter-efficient fine-tuning of models.
It assists in setting up adaptive configurations based on model size and available resources.

Usage:
    python src/lora_config.py --model XiaoduoAILab/XmodelLM1.5 --task qa
"""

import argparse
import json
import os
import logging
import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, TaskType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_model_size(model_name):
    """Estimate model size by loading model config"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, config_only=True)
        
        # Get number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        size_gb = num_params * 4 / (1024 ** 3)  # Assuming float32
        
        return {
            "name": model_name,
            "parameters": num_params,
            "size_gb": size_gb
        }
    except Exception as e:
        logger.warning(f"Could not determine model size: {e}")
        # Default to medium size if can't determine
        return {
            "name": model_name,
            "parameters": 1_000_000_000,  # 1B parameters
            "size_gb": 4.0
        }

def determine_target_modules(model_name):
    """Determine which modules should be targeted for LoRA adaptation"""
    # Default target modules for seq2seq models
    default_targets = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    
    # Customize based on known model architectures
    if "t5" in model_name.lower():
        return ["q", "v", "o"]
    elif "bart" in model_name.lower() or "mbart" in model_name.lower():
        return ["q_proj", "v_proj", "out_proj"]
    elif "xmodel" in model_name.lower():
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Default to common target modules
    return default_targets

def create_lora_config(model_name, task, rank=16, alpha=32, dropout=0.1):
    """Create LoRA configuration based on model and task"""
    # Get model information
    model_info = get_model_size(model_name)
    
    # Adjust rank based on model size
    if model_info["size_gb"] > 20:  # Very large model (20GB+)
        rank = min(rank, 8)  # Use smaller rank for very large models
    elif model_info["size_gb"] > 10:  # Large model (10-20GB)
        rank = min(rank, 16)
    elif model_info["size_gb"] < 2:  # Small model (<2GB)
        rank = max(rank, 32)  # Can use larger rank for small models
    
    # Determine target modules
    target_modules = determine_target_modules(model_name)
    
    # Determine task type
    if "causal" in model_name.lower() or "gpt" in model_name.lower() or "llama" in model_name.lower():
        task_type = TaskType.CAUSAL_LM
    else:
        task_type = TaskType.SEQ_2_SEQ_LM
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=task_type
    )
    
    # Generate config dict for saving
    config_dict = {
        "model_name": model_name,
        "task": task,
        "lora_config": {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": target_modules,
            "lora_dropout": dropout,
            "bias": "none",
            "task_type": str(task_type)
        },
        "model_info": model_info
    }
    
    return config_dict, lora_config

def save_config(config_dict, output_path):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"LoRA configuration saved to {output_path}")
    
    # Also save a human-readable summary
    summary_path = output_path.replace('.json', '.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"LoRA Configuration for {config_dict['model_name']}\n")
        f.write(f"Task: {config_dict['task']}\n\n")
        f.write(f"Model Parameters: {config_dict['model_info']['parameters']:,}\n")
        f.write(f"Model Size: {config_dict['model_info']['size_gb']:.2f} GB\n\n")
        f.write("LoRA Settings:\n")
        for key, value in config_dict['lora_config'].items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"LoRA configuration summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate LoRA configurations for MedMT")
    parser.add_argument('--model', type=str, default="XiaoduoAILab/XmodelLM1.5", 
                        help='Model name or path')
    parser.add_argument('--task', type=str, default="translation", 
                        choices=['translation', 'qa', 'summarization', 'reasoning'],
                        help='Task type')
    parser.add_argument('--rank', type=int, default=16, 
                        help='LoRA rank (r)')
    parser.add_argument('--alpha', type=int, default=32, 
                        help='LoRA alpha')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='LoRA dropout')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file path (default: lora_configs/{model}_{task}.json)')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Load project config for default values
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Use model from config if not specified
        if not args.model:
            args.model = config.get('pretrained_model', "XiaoduoAILab/XmodelLM1.5")
    except Exception as e:
        logger.warning(f"Could not load config file {args.config}: {e}")
    
    # Set default output path if not specified
    if not args.output:
        model_short_name = args.model.split('/')[-1]
        args.output = f"lora_configs/{model_short_name}_{args.task}.json"
    
    # Create and save LoRA config
    config_dict, lora_config = create_lora_config(
        args.model, 
        args.task, 
        rank=args.rank, 
        alpha=args.alpha, 
        dropout=args.dropout
    )
    
    save_config(config_dict, args.output)
    
    # Also save LoRA config as a Python script for easy import
    python_output = args.output.replace('.json', '.py')
    with open(python_output, 'w', encoding='utf-8') as f:
        f.write(f"""
from peft import LoraConfig, TaskType

# LoRA Config for {args.model} - {args.task}
LORA_CONFIG = LoraConfig(
    r={args.rank},
    lora_alpha={args.alpha},
    target_modules={config_dict['lora_config']['target_modules']},
    lora_dropout={args.dropout},
    bias="none",
    task_type=TaskType.{config_dict['lora_config']['task_type'].split('.')[-1]}
)
""")
    
    logger.info(f"LoRA configuration Python module saved to {python_output}")

if __name__ == "__main__":
    main()
