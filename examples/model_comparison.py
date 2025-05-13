"""
Compare translations between DeepSeek Reasoner and XiaoduoAILab/XmodelLM1.5 models.

This script takes a set of Chinese medical dialogues, translates them using both
models, and displays the results side by side for comparison.
"""
import os
import sys
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.load_xiaoduo_model import load_xiaoduo_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepSeekReasonerTranslator:
    """Wrapper for DeepSeek Reasoner model to provide translation functionality."""
    
    def __init__(self):
        """Initialize the DeepSeek Reasoner model for translation."""
        logger.info("Loading DeepSeek Reasoner model...")
        self.model_name = "deepseek-ai/deepseek-coder-33b-instruct"
        
        # Check for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with appropriate settings
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True
            )
        
        logger.info("DeepSeek Reasoner model loaded successfully")
    
    def translate_zh_to_th(self, chinese_text, temperature=0.7, max_new_tokens=512):
        """Translate Chinese text to Thai."""
        prompt = f"""You are a professional medical translator specialized in translating from Chinese to Thai.
Please translate the following Chinese medical dialogue to Thai, maintaining accuracy,
preserving medical terminology, and keeping the conversational flow natural.

Chinese dialogue:
{chinese_text}

Thai translation:"""
        
        logger.info("Generating translation with DeepSeek Reasoner...")
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the Thai translation
        translation = response.split("Thai translation:")[1].strip() if "Thai translation:" in response else response
        
        return translation

def compare_translations(dialogues, output_file=None):
    """
    Compare translations from DeepSeek Reasoner and XmodelLM1.5 models.
    
    Args:
        dialogues (list): List of Chinese dialogue strings
        output_file (str, optional): Path to save comparison results.
    """
    # Load models
    deepseek_translator = DeepSeekReasonerTranslator()
    logger.info("Loading XmodelLM1.5...")
    xmodel_translator = load_xiaoduo_model(use_instruct_version=True)
    
    if not xmodel_translator:
        logger.error("Failed to load XmodelLM1.5. Aborting comparison.")
        return
    
    # Run comparisons
    results = []
    logger.info(f"Comparing translations for {len(dialogues)} dialogues...")
    
    for i, dialogue in enumerate(dialogues):
        logger.info(f"Translating dialogue {i+1}/{len(dialogues)}")
        
        # Get translations from both models
        deepseek_translation = deepseek_translator.translate_zh_to_th(dialogue)
        xmodel_translation = xmodel_translator.translate_zh_to_th(dialogue)
        
        # Store results
        results.append({
            "source": dialogue,
            "deepseek": deepseek_translation,
            "xmodel": xmodel_translation
        })
        
        # Print comparison
        logger.info(f"\nOriginal Chinese:\n{dialogue}")
        logger.info(f"\nDeepSeek Reasoner:\n{deepseek_translation}")
        logger.info(f"\nXmodelLM1.5:\n{xmodel_translation}")
        logger.info("=" * 80)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                f.write(f"Dialogue {i+1}:\n\n")
                f.write(f"Chinese Source:\n{result['source']}\n\n")
                f.write(f"DeepSeek Reasoner Translation:\n{result['deepseek']}\n\n")
                f.write(f"XmodelLM1.5 Translation:\n{result['xmodel']}\n\n")
                f.write("=" * 80 + "\n\n")
        logger.info(f"Comparison results saved to {output_file}")
    
    return results

def main():
    # Sample Chinese medical dialogues
    sample_dialogues = [
        """医生：你最近血糖控制得怎么样？
病人：不太好，早餐后经常超过10。
医生：你的胰岛素用量是多少？
病人：每天早晚各10个单位。
医生：我建议你早餐前增加到12个单位，并且减少早餐的碳水化合物摄入。""",
        
        """医生：你头痛的情况持续多久了？
病人：大概三天了，主要是右侧太阳穴附近疼。
医生：疼痛是持续的还是间歇性的？
病人：间歇性的，但每次疼痛都很剧烈。
医生：你有没有伴随恶心、呕吐或者畏光的症状？
病人：有时候会有轻微恶心，但没有呕吐和畏光。"""
    ]
    
    # Output file path
    output_file = os.path.join(os.getcwd(), "model_comparison_results.txt")
    
    # Run comparison
    compare_translations(sample_dialogues, output_file)

if __name__ == "__main__":
    main()
