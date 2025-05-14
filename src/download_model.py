"""
Download and load the Salesforce BLIP-2 FLAN-T5-XL model.
"""
import os
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model():
    """
    Downloads and loads the Salesforce BLIP-2 FLAN-T5-XL model.
    Returns a wrapper class containing the model and processor.
    """
    repo_id = "Salesforce/blip2-flan-t5-xl"
    local_path = "./MedicalMT_/models/blip2-flan-t5-xl"
    
    logger.info(f"Attempting to load model: {repo_id}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)
        
        # Load processor and model
        processor = Blip2Processor.from_pretrained(repo_id)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model
        model = Blip2ForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        logger.info(f"Successfully loaded model on {device}")
        
        # Create wrapper class
        class ModelWrapper:
            def __init__(self, model, processor, device):
                self.model = model 
                self.processor = processor
                self.device = device
            
            def generate(self, image_path, prompt, max_new_tokens=100):
                """Generate text based on image and prompt"""
                # Load and preprocess image
                from PIL import Image
                image = Image.open(image_path)
                
                # Prepare inputs
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )
                
                # Decode and return generated text
                return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Create and return wrapper
        wrapper = ModelWrapper(model, processor, device)
        logger.info("Model wrapper created successfully")
        
        return wrapper
        
    except Exception as e:
        logger.error(f"An error occurred while loading the model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    model = load_model()
    
    if model:
        logger.info("Model loaded successfully!")
    else:
        logger.error("Failed to load model.")

if __name__ == "__main__":
    main()
