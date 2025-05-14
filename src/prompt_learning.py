"""
Prompt/Context Learning for MedMT

This script implements few-shot/prompt-based learning for medical translation.
It enhances translation quality by leveraging context and in-context examples.

Usage:
    python src/prompt_learning.py --model models/MedicalZbitxLM --input data/test.csv --output results/prompt_results.csv
"""

import argparse
import os
import yaml
import logging
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

class PromptDesigner:
    """Class for generating and managing translation prompts"""
    
    def __init__(self, context_examples=None, max_examples=5):
        self.context_examples = context_examples or []
        self.max_examples = max_examples
        self.templates = self._get_templates()
    
    def _get_templates(self):
        """Define prompt templates for different scenarios"""
        templates = {
            "simple": (
                "Translate the following Chinese medical text to Thai. "
                "Use the context provided before the text.\n\n"
                "{examples}\n\n"
                "Context: {context}\n"
                "Chinese: {source}\n"
                "Thai:"
            ),
            
            "medical": (
                "You are a medical translator specializing in Chinese to Thai translation. "
                "Translate the following Chinese medical text to Thai with careful attention to medical terminology.\n\n"
                "{examples}\n\n"
                "Context: {context}\n"
                "Chinese: {source}\n"
                "Thai:"
            ),
            
            "conversational": (
                "This is a medical conversation between a doctor and patient. "
                "Translate the following Chinese dialogue to natural-sounding Thai.\n\n"
                "{examples}\n\n"
                "Context: {context}\n"
                "Chinese: {source}\n"
                "Thai:"
            ),
            
            "few_shot": (
                "Below are examples of Chinese to Thai medical translations. "
                "Use these examples to guide your translation of the new text.\n\n"
                "{examples}\n\n"
                "Now translate:\n"
                "Context: {context}\n"
                "Chinese: {source}\n"
                "Thai:"
            ),
            
            "formal": (
                "Please provide a formal Thai translation of the following Chinese medical text. "
                "Maintain the professional tone appropriate for a medical setting.\n\n"
                "{examples}\n\n"
                "Context: {context}\n"
                "Chinese: {source}\n"
                "Thai:"
            )
        }
        return templates
    
    def format_examples(self, examples=None):
        """Format examples for inclusion in prompts"""
        examples_to_use = examples if examples is not None else self.context_examples
        
        if not examples_to_use:
            return ""
        
        # Limit to max_examples
        if len(examples_to_use) > self.max_examples:
            examples_to_use = random.sample(examples_to_use, self.max_examples)
        
        formatted = ""
        for example in examples_to_use:
            context = example.get('context', '')
            source = example.get('source', '')
            target = example.get('target', '')
            
            formatted += f"Context: {context}\nChinese: {source}\nThai: {target}\n\n"
        
        return formatted
    
    def get_prompt(self, template_name, context, source, examples=None):
        """Get a formatted prompt using the specified template"""
        if template_name not in self.templates:
            logger.warning(f"Template '{template_name}' not found. Using 'simple' template.")
            template_name = "simple"
        
        template = self.templates[template_name]
        formatted_examples = self.format_examples(examples)
        
        return template.format(
            examples=formatted_examples,
            context=context,
            source=source
        )
    
    def get_all_prompts(self, context, source, examples=None):
        """Get all available prompts for a given input"""
        return {
            name: self.get_prompt(name, context, source, examples)
            for name in self.templates
        }

def load_examples(file_path, n_examples=5):
    """Load examples for few-shot learning"""
    try:
        df = pd.read_csv(file_path)
        # Select random examples
        if len(df) > n_examples:
            examples = df.sample(n_examples).to_dict('records')
        else:
            examples = df.to_dict('records')
        
        logger.info(f"Loaded {len(examples)} examples from {file_path}")
        return examples
    except Exception as e:
        logger.error(f"Failed to load examples from {file_path}: {e}")
        return []

def evaluate_prompts(model, tokenizer, test_df, prompt_designer, device):
    """Evaluate different prompt templates"""
    results = {}
    
    # Get first few examples from test data if they have target (golden answer)
    examples = None
    if 'target' in test_df.columns:
        examples = test_df.head(5).to_dict('records')
    
    # Test each template on a subset of data
    eval_df = test_df.head(10)  # Use first 10 samples for quick evaluation
    
    for template_name in prompt_designer.templates:
        logger.info(f"Evaluating template: {template_name}")
        predictions = []
        
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Template {template_name}"):
            context = row.get('context', '')
            source = row.get('source', '')
            
            prompt = prompt_designer.get_prompt(template_name, context, source, examples)
            
            # Generate prediction
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=256,
                    num_beams=4,
                    temperature=0.7,
                )
            
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            predictions.append(prediction)
        
        # Calculate simple metrics (avg length)
        avg_length = sum(len(p) for p in predictions) / len(predictions) if predictions else 0
        results[template_name] = {
            'predictions': predictions,
            'avg_length': avg_length
        }
    
    # Select best template based on average output length
    # This is just a simple heuristic; better metrics could be used if reference translations are available
    best_template = max(results.items(), key=lambda x: x[1]['avg_length'])[0]
    logger.info(f"Selected best template: {best_template}")
    
    return best_template, results

def translate_with_prompt(model_path, input_file, output_file, examples_file=None, template=None, n_examples=5):
    """Translate using prompt/context-based approach"""
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Try using MedMTModel
        model = MedMTModel(model_path)
        tokenizer = model.tokenizer
        base_model = model.model
    except Exception as e:
        logger.warning(f"Failed to load model using MedMTModel: {e}")
        logger.info("Trying to load model directly with transformers...")
        
        # Direct loading with transformers
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    
    base_model.to(device)
    base_model.eval()
    
    # Load data
    if input_file.endswith('.csv'):
        test_df = pd.read_csv(input_file)
    else:
        logger.error(f"Unsupported file format: {input_file}")
        return
    
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # Load examples for few-shot learning
    examples = None
    if examples_file:
        examples = load_examples(examples_file, n_examples)
    
    # Initialize prompt designer
    prompt_designer = PromptDesigner(context_examples=examples, max_examples=n_examples)
    
    # Determine best template if not specified
    if not template:
        logger.info("No template specified. Evaluating templates...")
        template, _ = evaluate_prompts(base_model, tokenizer, test_df, prompt_designer, device)
    
    # Translate using the selected template
    logger.info(f"Translating with template: {template}")
    predictions = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Translating"):
        context = row.get('context', '')
        source = row.get('source', '')
        
        prompt = prompt_designer.get_prompt(template, context, source, examples)
        
        # Generate prediction
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            output = base_model.generate(
                input_ids,
                max_length=256,
                num_beams=4,
                temperature=0.7,
            )
        
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(prediction)
    
    # Save results
    result_df = test_df.copy()
    result_df['prediction'] = predictions
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    # Also save the prompt template used
    template_file = os.path.join(os.path.dirname(output_file), 'prompt_template.txt')
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(f"Template: {template}\n\n")
        f.write(prompt_designer.templates[template])
    
    logger.info(f"Prompt template saved to {template_file}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Prompt/Context Learning for MedMT")
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to model directory or name')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input file (CSV)')
    parser.add_argument('--output', type=str, required=True, 
                        help='Output file for predictions (CSV)')
    parser.add_argument('--examples', type=str, default=None, 
                        help='Examples file for few-shot learning (CSV)')
    parser.add_argument('--template', type=str, default=None, 
                        choices=['simple', 'medical', 'conversational', 'few_shot', 'formal'],
                        help='Prompt template to use (optional)')
    parser.add_argument('--n_examples', type=int, default=5, 
                        help='Number of examples to use for few-shot learning')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to config file (optional)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load config if specified
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Get model path from config if not specified
    if not args.model:
        args.model = config.get('model_save_path', None)
        if not args.model:
            logger.error("No model specified and none found in config")
            return
    
    # Run translation with prompt learning
    translate_with_prompt(
        model_path=args.model,
        input_file=args.input,
        output_file=args.output,
        examples_file=args.examples,
        template=args.template,
        n_examples=args.n_examples
    )

if __name__ == "__main__":
    main()
