"""
Load the XiaoduoAILab/XmodelLM1.5 model using its custom modeling files.

This script is a replacement for load_xmodel_example.py that properly handles
the custom model type and configuration of the XmodelLM1.5 model.
"""
import torch
import os
import sys
import importlib
import importlib.util
from huggingface_hub import snapshot_download
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_module_from_file(file_path, module_name):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def fix_relative_imports(model_dir):
    """Fix relative imports by making files importable from each other."""
    # Create empty __init__.py in model directory to make it a package
    init_path = os.path.join(model_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            pass  # Create empty file
    
    # Add the parent directory to sys.path so the model dir is importable
    parent_dir = os.path.dirname(model_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Make the model dir itself importable
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
        
    return os.path.basename(model_dir)

def load_xiaoduo_model(use_instruct_version=True):
    """
    Downloads the XiaoduoAILab/XmodelLM1.5 model to a local directory,
    loads it using the model's custom modeling files, and sets it up for use.
    
    Args:
        use_instruct_version (bool): Whether to use the instruction-tuned version (recommended)
                                    or the pretrained base model.
    """
    repo_id = "XiaoduoAILab/XmodelLM1.5"
    local_snapshot_path = "./XiaoduoAILab/XmodelLM1.5"
    
    logger.info(f"Attempting to download model: {repo_id} to {os.path.abspath(local_snapshot_path)}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_snapshot_path,
            resume_download=True
        )
        logger.info(f"Model downloaded successfully to {os.path.abspath(local_snapshot_path)}")
    except Exception as e:
        logger.error(f"An error occurred during model download: {e}")
        return None
      # Select the correct subdirectory based on whether to use instruct or pretrain
    subdir = "instruct" if use_instruct_version else "pretrain"
    model_path = os.path.join(local_snapshot_path, subdir)
    
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Using {'instruct' if use_instruct_version else 'pretrain'} version")
    
    try:
        # Fix relative imports by making the model directory a package
        package_name = fix_relative_imports(model_path)
        
        # Load the custom model implementation files
        modeling_file = os.path.join(model_path, "modeling_xmodel.py")
        config_file = os.path.join(model_path, "configuration_xmodel.py")
        tokenizer_file = os.path.join(model_path, "tokenization_xmodel.py")
          # Verify files exist
        for file_path in [modeling_file, config_file, tokenizer_file]:
            if not os.path.exists(file_path):
                logger.error(f"Required file not found: {file_path}")
                return None
        
        # Import the modules using import statement now that we've made them importable
        module_prefix = f"{package_name}."
        
        # First import the configuration
        config_module_name = f"{module_prefix}configuration_xmodel"
        config_module = importlib.import_module(config_module_name)
        
        # Then import tokenizer and modeling which might depend on configuration
        tokenizer_module_name = f"{module_prefix}tokenization_xmodel"
        tokenizer_module = importlib.import_module(tokenizer_module_name)
        modeling_module_name = f"{module_prefix}modeling_xmodel"
        modeling_module = importlib.import_module(modeling_module_name)
        
        # Get the model configuration
        config_class = getattr(config_module, "XModelConfig")  # Note: "M" is capitalized
        config = config_class.from_pretrained(model_path)
        
        # Get the tokenizer
        tokenizer_class = getattr(tokenizer_module, "XModelTokenizer")
        tokenizer = tokenizer_class.from_pretrained(model_path)
        
        # Get the model class
        if use_instruct_version:
            model_class = getattr(modeling_module, "XModelForCausalLM")
        else:
            model_class = getattr(modeling_module, "XModel")
        
        # Load the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model = model_class.from_pretrained(
            model_path,
            config=config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        logger.info(f"Successfully loaded model on {device}")
        
        # Create a simple wrapper class to make the model easier to use
        class XmodelWrapper:
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
            
            def generate(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
                """Generate text based on a prompt"""
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True
                    )
                
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            def translate_zh_to_th(self, chinese_text, system_prompt=None):
                """Translate Chinese text to Thai"""
                if system_prompt is None:
                    system_prompt = (
                        "You are a medical translator specialized in Chinese to Thai translation. "
                        "Translate the provided Chinese text to Thai accurately, "
                        "preserving medical terminology and natural language flow."
                    )
                
                prompt = f"{system_prompt}\n\nChinese text: {chinese_text}\n\nThai translation:"
                response = self.generate(prompt)
                
                # Extract just the Thai translation from the response
                # This is a simple approach - you may need more sophisticated extraction
                translation = response.split("Thai translation:")[1].strip() if "Thai translation:" in response else response
                
                return translation
        
        # Create and return the wrapper
        wrapper = XmodelWrapper(model, tokenizer, device)
        logger.info("Wrapper created successfully")
        
        # Test the model with a simple example
        logger.info("Testing the model with a simple prompt...")
        test_prompt = "Hello, how are you?"
        test_output = wrapper.generate(test_prompt, max_new_tokens=50)
        logger.info(f"Test output: {test_output}")
        
        return wrapper
    
    except Exception as e:
        logger.error(f"An error occurred while loading the model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    model = load_xiaoduo_model(use_instruct_version=True)
    
    if model:
        logger.info("Model loaded successfully!")
        
        # Test Chinese to Thai translation
        zh_text = "医生：你最近血糖控制得怎么样？"
        logger.info(f"Testing Chinese to Thai translation with text: {zh_text}")
        
        th_translation = model.translate_zh_to_th(zh_text)
        logger.info(f"Translation: {th_translation}")
    else:
        logger.error("Failed to load the model.")

if __name__ == "__main__":
    main()
