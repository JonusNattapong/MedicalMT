# Using XiaoduoAILab/XmodelLM1.5 in MedMT

This guide provides detailed instructions on how to use the XiaoduoAILab/XmodelLM1.5 model for Chinese to Thai medical translation in the MedMT project.

## Overview

The XiaoduoAILab/XmodelLM1.5 model is a multilingual model with strong capabilities for Asian language translation. In the MedMT project, we've implemented a custom loading mechanism to properly use this model for Chinese to Thai medical translation.

## Prerequisites

Before using the model, ensure you have:

1. Sufficient disk space (~6GB for the model)
2. PyTorch installed
3. Internet connection for initial download

## Basic Usage

### Loading the Model

```python
from src.load_xiaoduo_model import load_xiaoduo_model

# Load the instruction-tuned version (recommended for translation tasks)
model = load_xiaoduo_model(use_instruct_version=True)

# For basic translation
chinese_text = "医生：你最近血糖控制得怎么样？"
thai_translation = model.translate_zh_to_th(chinese_text)
print(f"Translation: {thai_translation}")
```

### Customizing Translation

You can customize the translation by providing a system prompt:

```python
system_prompt = (
    "You are a medical translator specialized in diabetes terminology. "
    "Translate the provided Chinese text to Thai accurately, "
    "ensuring all diabetes-related terms are correctly translated."
)

chinese_text = "医生：你需要每天监测血糖，注意胰岛素用量。"
thai_translation = model.translate_zh_to_th(chinese_text, system_prompt=system_prompt)
```

### Advanced Parameters

When using the model for generation, you can customize these parameters:

```python
# Using the generate method directly
response = model.generate(
    prompt="Translate to Thai: 医生：你好，请问有什么不舒服？",
    max_new_tokens=512,  # Controls the maximum length of the generated text
    temperature=0.7,     # Controls randomness (higher = more random)
    top_p=0.9            # Controls diversity of responses
)
```

## Example Scripts

The `examples/` directory contains several scripts for using the model:

1. **Basic Translation Example**:
   ```bash
   python examples/xiaoduo_translation_example.py
   ```

2. **Comparing with DeepSeek Reasoner**:
   ```bash
   python examples/model_comparison.py
   ```

3. **Evaluating Translation Quality**:
   ```bash
   python examples/evaluate_translations.py --dataset data/your_test_data.csv
   ```

## Model Details

- **Model Name**: XiaoduoAILab/XmodelLM1.5
- **Model Type**: Custom implementation (not standard Hugging Face format)
- **Size**: ~6GB
- **Languages**: Multilingual with strength in Asian languages
- **Loading Method**: Custom implementation using the model's own modeling files

## Troubleshooting

### Common Issues

1. **"No module named 'modeling_xmodel'"**:
   - This is normal. Our implementation dynamically loads the model's custom modules.

2. **High memory usage**:
   - The model requires approximately 6GB of RAM. For better performance, use a GPU.

3. **Slow first inference**:
   - The first translation might be slower as the model loads into memory.

### Memory Optimization

For lower memory usage, you can try:

```python
from src.load_xiaoduo_model import load_xiaoduo_model
import torch

# Force CPU usage for lower memory systems
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load model with lower precision
model = load_xiaoduo_model(use_instruct_version=True)
```

## License Considerations

The XiaoduoAILab/XmodelLM1.5 model has its own license terms. Please refer to the model's Hugging Face page for details on usage restrictions.

## Performance Comparison

Initial testing shows that the XmodelLM1.5 model performs particularly well for:

1. Preserving medical terminology
2. Natural-sounding Thai translations
3. Handling conversational context

However, DeepSeek Reasoner may perform better for:

1. Complex medical explanations
2. Highly technical content
3. Longer passages with intricate context

Choose the appropriate model based on your specific translation needs.
