# MedMT Examples

This directory contains example scripts demonstrating how to use the MedMT project for Chinese-to-Thai medical dialogue translation with different models.

## Available Examples

### 1. XiaoduoAILab/XmodelLM1.5 Translation Example

This example shows how to use the XiaoduoAILab/XmodelLM1.5 model for translating Chinese medical dialogues to Thai.

```bash
python examples/xiaoduo_translation_example.py
```

The script will:
- Load the XmodelLM1.5 model using our custom implementation
- Translate sample medical dialogues from Chinese to Thai
- Save the translations to `xmodel_translation_examples.txt`

### 2. Model Comparison

This example compares translations from DeepSeek Reasoner and XiaoduoAILab/XmodelLM1.5 models.

```bash
python examples/model_comparison.py
```

The script will:
- Load both DeepSeek Reasoner and XmodelLM1.5 models
- Translate the same sample dialogues with both models
- Compare the translation outputs side by side
- Save the comparison results to `model_comparison_results.txt`

## Notes on Model Usage

### DeepSeek Reasoner

- Uses the `deepseek-ai/deepseek-coder-33b-instruct` model
- Requires more memory but provides high-quality translations
- Better at handling complex medical terminology

### XiaoduoAILab/XmodelLM1.5

- Custom model implementation for multilingual translation
- More lightweight than DeepSeek Reasoner
- Specialized in Asian language translation

## Example Medical Dialogues

The example scripts include sample Chinese medical dialogues covering:
1. Diabetes management
2. Headache symptoms

You can modify these examples or add your own dialogues by editing the `sample_dialogues` list in each script.

## Adding New Examples

To create a new example script:

1. Create a new Python file in the `examples` directory
2. Import the necessary models from `src/`
3. Add your custom translation logic
4. Document your example in this README

## Runtime Requirements

- For DeepSeek Reasoner: CUDA-capable GPU with at least 16GB VRAM recommended
- For XmodelLM1.5: Can run on CPU, but GPU recommended for faster inference
- Python 3.8+ with PyTorch installed
