"""
Comprehensive test for the XiaoduoAILab/XmodelLM1.5 model.

This script performs a series of tests to verify that the model
is loading correctly and can perform various translation tasks.
"""
import os
import sys
import logging
import torch
import time
from pathlib import Path

# Add the src directory to the path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

try:
    from load_xiaoduo_model import load_xiaoduo_model
except ImportError:
    print("Error: Could not import load_xiaoduo_model. Make sure you're running from the project root.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', f'xmodel_test_{time.strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading the model with different configurations."""
    logger.info("=== Testing Model Loading ===")
    
    # Test 1: Load instruction-tuned model
    logger.info("Test 1: Loading instruction-tuned model...")
    start_time = time.time()
    model_instruct = load_xiaoduo_model(use_instruct_version=True)
    load_time = time.time() - start_time
    
    if model_instruct:
        logger.info(f"✓ Successfully loaded instruction-tuned model in {load_time:.2f} seconds")
    else:
        logger.error("✗ Failed to load instruction-tuned model")
        return False
    
    # Test 2: Load pretrained model
    logger.info("Test 2: Loading pretrained model...")
    start_time = time.time()
    model_pretrain = load_xiaoduo_model(use_instruct_version=False)
    load_time = time.time() - start_time
    
    if model_pretrain:
        logger.info(f"✓ Successfully loaded pretrained model in {load_time:.2f} seconds")
    else:
        logger.error("✗ Failed to load pretrained model")
    
    # Return the instruction-tuned model for further testing
    return model_instruct

def test_basic_generation(model):
    """Test basic text generation capabilities."""
    logger.info("=== Testing Basic Generation ===")
    
    # Test prompts
    prompts = [
        "Translate to Thai: Hello, how are you?",
        "Write a short poem about nature.",
        "Explain what is diabetes in simple terms."
    ]
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Test {i+1}: Generating text for prompt: '{prompt}'")
        
        try:
            start_time = time.time()
            response = model.generate(prompt, max_new_tokens=100)
            gen_time = time.time() - start_time
            
            logger.info(f"✓ Generation completed in {gen_time:.2f} seconds")
            logger.info(f"Response: {response}")
        except Exception as e:
            logger.error(f"✗ Generation failed: {e}")
            return False
    
    return True

def test_translation(model):
    """Test Chinese to Thai translation capabilities."""
    logger.info("=== Testing Chinese to Thai Translation ===")
    
    # Test translations with different complexities
    test_cases = [
        # Simple greeting
        {
            "zh": "你好，今天感觉怎么样？",
            "description": "Simple greeting"
        },
        # Medical terminology
        {
            "zh": "病人血压偏高，需要定期监测并服用降压药。",
            "description": "Medical terminology"
        },
        # Dialogue
        {
            "zh": "医生：你最近头痛的情况如何？\n病人：还是时不时会疼，尤其是晚上。",
            "description": "Medical dialogue"
        },
        # Complex sentence
        {
            "zh": "根据最新的临床指南，二型糖尿病患者应该每三个月检查一次糖化血红蛋白，并根据检查结果调整用药。",
            "description": "Complex medical sentence"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        zh_text = test_case["zh"]
        description = test_case["description"]
        
        logger.info(f"Test {i+1}: Translating '{description}'")
        logger.info(f"Source: {zh_text}")
        
        try:
            start_time = time.time()
            translation = model.translate_zh_to_th(zh_text)
            trans_time = time.time() - start_time
            
            logger.info(f"✓ Translation completed in {trans_time:.2f} seconds")
            logger.info(f"Translation: {translation}")
        except Exception as e:
            logger.error(f"✗ Translation failed: {e}")
            return False
    
    return True

def test_custom_parameters(model):
    """Test model with custom parameters."""
    logger.info("=== Testing Custom Parameters ===")
    
    test_text = "医生：你有什么不舒服的症状？"
    
    parameter_sets = [
        {"temperature": 0.5, "max_new_tokens": 100},
        {"temperature": 0.9, "max_new_tokens": 50},
        {"temperature": 0.7, "top_p": 0.8, "max_new_tokens": 200}
    ]
    
    for i, params in enumerate(parameter_sets):
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        logger.info(f"Test {i+1}: Translating with parameters: {param_str}")
        
        try:
            start_time = time.time()
            translation = model.translate_zh_to_th(test_text)
            trans_time = time.time() - start_time
            
            logger.info(f"✓ Translation completed in {trans_time:.2f} seconds")
            logger.info(f"Translation: {translation}")
        except Exception as e:
            logger.error(f"✗ Translation with custom parameters failed: {e}")
            return False
    
    return True

def test_custom_prompts(model):
    """Test model with custom system prompts."""
    logger.info("=== Testing Custom System Prompts ===")
    
    test_text = "医生：你的血糖水平有些高，需要控制饮食和增加运动。"
    
    system_prompts = [
        "You are a medical translator specializing in diabetes terminology. Translate the Chinese text to Thai.",
        "You are a helpful assistant that translates medical conversations from Chinese to Thai accurately.",
        "Translate the following Chinese medical text to Thai, maintaining formal medical language."
    ]
    
    for i, prompt in enumerate(system_prompts):
        logger.info(f"Test {i+1}: Translating with custom system prompt")
        logger.info(f"System prompt: {prompt}")
        
        try:
            start_time = time.time()
            translation = model.translate_zh_to_th(test_text, system_prompt=prompt)
            trans_time = time.time() - start_time
            
            logger.info(f"✓ Translation completed in {trans_time:.2f} seconds")
            logger.info(f"Translation: {translation}")
        except Exception as e:
            logger.error(f"✗ Translation with custom system prompt failed: {e}")
            return False
    
    return True

def generate_test_report(results):
    """Generate a test report with the results."""
    logger.info("=== Generating Test Report ===")
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
      # Generate report file path
    report_path = os.path.join(logs_dir, f'xmodel_test_report_{time.strftime("%Y%m%d_%H%M%S")}.md')
      # Check if all tests passed
    all_passed = all(results.values())
    
    # Force status to FAILED if Model Loading failed
    if 'Model Loading' in results and not results['Model Loading']:
        all_passed = False
        status = "❌ FAILED"
    else:
        status = "✅ PASSED" if all_passed else "❌ FAILED"
    
    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# XiaoduoAILab/XmodelLM1.5 Test Report\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Overall Status: {status}\n\n")
        f.write("## Test Results\n\n")
        
        for test_name, passed in results.items():
            result = "✅ PASSED" if passed else "❌ FAILED"
            f.write(f"- {test_name}: {result}\n")
        
        f.write("\n## System Information\n\n")
        f.write(f"- Python version: {sys.version}\n")
        f.write(f"- PyTorch version: {torch.__version__}\n")
        f.write(f"- CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"- CUDA version: {torch.version.cuda}\n")
            f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
        
        f.write("\n## Notes\n\n")
        if all_passed:
            f.write("All tests passed successfully. The XmodelLM1.5 model is working correctly.\n")
        else:
            f.write("Some tests failed. Please check the log for details.\n")
    
    logger.info(f"Test report generated: {report_path}")
    return report_path

def main():
    logger.info("Starting comprehensive test of XiaoduoAILab/XmodelLM1.5 model")
    
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Dictionary to store test results
    results = {}
    
    # Test 1: Model Loading
    model = test_model_loading()
    results["Model Loading"] = model is not None
    
    if not model:
        logger.error("Model loading failed. Cannot continue with further tests.")
        report_path = generate_test_report(results)
        
        # Print summary
        logger.warning("⚠️ Some tests failed. Check the report for details.")
        logger.info(f"Test report available at: {report_path}")
        return
    
    # Test 2: Basic Generation
    results["Basic Generation"] = test_basic_generation(model)
    
    # Test 3: Translation
    results["Translation"] = test_translation(model)
    
    # Test 4: Custom Parameters
    results["Custom Parameters"] = test_custom_parameters(model)
    
    # Test 5: Custom Prompts
    results["Custom Prompts"] = test_custom_prompts(model)
    
    # Generate test report
    report_path = generate_test_report(results)
    
    # Print summary
    all_passed = all(results.values())
    if all_passed:
        logger.info("🎉 All tests passed successfully!")
    else:
        logger.warning("⚠️ Some tests failed. Check the report for details.")
    
    logger.info(f"Test report available at: {report_path}")

if __name__ == "__main__":
    main()
