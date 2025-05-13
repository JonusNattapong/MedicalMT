"""
Example script for translating a medical dialogue from Chinese to Thai using
the XiaoduoAILab/XmodelLM1.5 model.

This script shows how to use the model for medical translations
after loading it with the load_xiaoduo_model.py implementation.
"""
import os
import sys
import logging
from src.load_xiaoduo_model import load_xiaoduo_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def translate_medical_dialogues(dialogues, output_file=None):
    """
    Translate a list of Chinese medical dialogues to Thai using the XmodelLM1.5 model.
    
    Args:
        dialogues (list): List of Chinese dialogue strings
        output_file (str, optional): Path to save translations. If None, only prints results.
    
    Returns:
        list: List of Thai translations
    """
    # Load the model
    logger.info("Loading XmodelLM1.5 model...")
    model = load_xiaoduo_model(use_instruct_version=True)
    
    if not model:
        logger.error("Failed to load the model.")
        return []
    
    # Custom prompt for medical dialogue translation
    system_prompt = (
        "You are a medical translator specialized in Chinese to Thai translation. "
        "Translate the provided Chinese medical dialogue to Thai accurately, "
        "preserving medical terminology and the natural conversational flow between doctor and patient. "
        "Maintain the same speaker indicators (Doctor/Patient) in the Thai translation."
    )
    
    # Translate each dialogue
    translations = []
    logger.info(f"Translating {len(dialogues)} dialogues...")
    
    for i, dialogue in enumerate(dialogues):
        logger.info(f"Translating dialogue {i+1}/{len(dialogues)}")
        translation = model.translate_zh_to_th(dialogue, system_prompt=system_prompt)
        translations.append(translation)
        
        # Print the results
        logger.info(f"\nOriginal Chinese:\n{dialogue}")
        logger.info(f"\nThai Translation:\n{translation}")
        logger.info("-" * 50)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (src, tgt) in enumerate(zip(dialogues, translations)):
                f.write(f"Dialogue {i+1}:\n")
                f.write(f"Chinese:\n{src}\n\n")
                f.write(f"Thai:\n{tgt}\n")
                f.write("=" * 50 + "\n\n")
        logger.info(f"Translations saved to {output_file}")
    
    return translations

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
    output_file = os.path.join(os.getcwd(), "xmodel_translation_examples.txt")
    
    # Run translation
    translate_medical_dialogues(sample_dialogues, output_file)

if __name__ == "__main__":
    main()
