# MedMT: Medical Dialogue Machine Translation (Chinese→Thai)

<p align="center" style="position: relative;">
    <img src="assets/logo.png" alt="MedMT Logo" width="200" style="position: relative; z-index: 1;"/>
</p>

<p align="center">
    <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
        <img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License: CC BY-NC-SA 4.0"/>
    </a>
</p>

## 📖 Overview

MedMT is a specialized machine translation system for medical dialogues between Chinese and Thai languages. Using advanced AI technologies and context-aware translation, it aims to preserve both medical accuracy and conversational nuances.

## 🆕 Latest Updates

- **Multi-turn Dialogue Generation**: New mode for generating realistic multi-turn conversations between doctors and patients
- **Automatic Quality Assessment**: Built-in tools for translation quality evaluation
- **Response Caching System**: Improved efficiency through API response caching
- **Dataset Analysis Tools**: Comprehensive analysis of dataset diversity and quality metrics
- **DeepSeek Reasoner Integration**: Enhanced dataset generation using DeepSeek Reasoner model for higher quality translations
- **XmodelLM1.5 Support**: Added custom implementation for loading and using the XiaoduoAILab/XmodelLM1.5 model
- **Improved Dataset Diversity**: Enhanced prompt engineering and parameter strategies for more diverse medical dialogues
- **Model Comparison Tools**: Added scripts to compare translations between different models

## ✨ Key Features

- **Context-aware Translation**: Leverages multi-turn dialogue context
- **Medical Accuracy**: Preserves medical terminology and meaning
- **Synthetic Data**: High-quality synthetic dialogues using DeepSeek AI
- **Multiple Formats**: Supports dialogue, Q&A, and multi-turn conversation formats
- **Quality Assessment**: Automatic evaluation of translation quality
- **Dataset Analysis**: Tools to analyze dataset diversity and characteristics
- **Advanced Training**: LoRA/PEFT for efficient fine-tuning
- **Scalable**: DeepSpeed/Accelerate for multi-GPU training
- **Multi-Model Support**: Integration with DeepSeek Reasoner and XiaoduoAILab/XmodelLM1.5

## 🛠️ Components

### 1. Data Generation

- Uses DeepSeek Reasoner for enhanced synthetic data generation
- Supports dialogue, Q&A, and multi-turn conversation formats
- Includes automatic quality assessment
- Response caching for improved efficiency
- Dataset analysis tools for quality control
- Includes metadata and licensing information
- Licensed under CC BY-SA-NC 4.0

### 2. Model Development

- Based on multiple pretrained models (mBART, NLLB, XmodelLM1.5)
- Fine-tuned with LoRA/PEFT
- Scaled with DeepSpeed/Accelerate
- Context-aware translation capabilities

### 3. Evaluation System

- Automatic metrics evaluation (BLEU, METEOR)
- Human evaluation support
- Quality assurance checks
- Model comparison tools

### 4. Deployment Options

- Hugging Face Hub integration
- Local inference capabilities
- Easy-to-use API
- Custom model loading for specialized models

## 🚀 Getting Started

### System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 16GB RAM minimum (32GB recommended for training)
- 100GB disk space for datasets and models

### Dependencies

```bash
# Core ML libraries
torch>=1.13
transformers==4.40.0
datasets>=2.19.0
sentencepiece
sacrebleu

# Training utilities
accelerate
peft
deepspeed
safetensors
huggingface_hub

# Data processing
pandas
scikit-learn
tqdm
openpyxl

# Development
jupyter
```

### Installation

```bash
git clone https://github.com/JonusNattapong/medmt.git
cd medmt
pip install -r requirements.txt
```

### Generate Dataset

```bash
# Generate dialogue dataset
python src/generate_dataset.py --output data/dialogue_train.csv --n_samples 1000 --mode dialogue

# Generate QA dataset
python src/generate_dataset.py --output data/qa_train.csv --n_samples 1000 --mode qa

# Generate multi-turn dialogue dataset
python src/generate_dataset.py --output data/multiturn_train.csv --n_samples 50 --mode multiturn

# Generate dataset with quality assessment
python src/generate_dataset.py --output data/qa_train.csv --n_samples 1000 --mode qa --assess_quality

# Use response caching to improve efficiency
python src/generate_dataset.py --output data/dialogue_train.csv --n_samples 1000 --use_cache

# Analyze an existing dataset
python src/analyze_dataset.py data/dialogue_train.csv --output-dir data/analysis
```

### Training

```bash
# Basic training
python src/train.py --config config.yaml

# Training with DeepSpeed
deepspeed src/train.py --config config.yaml --deepspeed ds_config.json

# Training with LoRA
python src/train.py --config config.yaml --use_lora
```

### Evaluation

```bash
python src/evaluate.py --config config.yaml
```

### Inference

```bash
python src/inference.py --input data/test.csv --output predictions.csv
```

### Using XiaoduoAILab/XmodelLM1.5

```bash
# Run a simple translation example
python examples/xiaoduo_translation_example.py

# Compare translations between DeepSeek Reasoner and XmodelLM1.5
python examples/model_comparison.py

# Evaluate translation quality against a reference dataset
python examples/evaluate_translations.py --dataset data/test_data.csv --output evaluation_results.csv
```

For more details on using the XmodelLM1.5 model, see the `examples/README.md` file.

## 📊 Dataset Format

### Dialogue Format

```python
{
    "context": "医生：你最近血糖控制得怎么样？\n病人：有点高。",
    "source": "你最近饮食有注意吗？",
    "target": "ช่วงนี้คุณควบคุมอาหารดีไหม?"
}
```

### QA Format

```python
{
    "context": "患者最近血糖偏高，且有口渴、多尿等症状。",
    "question_zh": "这可能是什么疾病？需要做什么检查？",
    "answer_zh": "从症状来看，可能是糖尿病。建议做空腹血糖和糖化血红蛋白检查。",
    "question_th": "อาการเหล่านี้อาจเป็นโรคอะไร? ควรตรวจอะไรเพิ่มเติม?",
    "answer_th": "จากอาการที่พบ อาจเป็นโรคเบาหวาน แนะนำให้ตรวจน้ำตาลในเลือดขณะอดอาหารและค่า HbA1c"
}
```

### Multi-turn Dialogue Format

```python
{
    "context": "患者有头痛症状，需要医疗建议。",
    "topic_zh": "头痛",
    "topic_th": "ปวดศีรษะ",
    "turn_number": 1,
    "patient_zh": "我最近经常头痛，特别是早上起床后。",
    "patient_th": "ช่วงนี้ฉันปวดหัวบ่อย โดยเฉพาะหลังตื่นนอนตอนเช้า",
    "doctor_zh": "您的头痛是持续性的还是阵发性的？疼痛程度如何？",
    "doctor_th": "อาการปวดหัวของคุณเป็นแบบต่อเนื่องหรือเป็นพักๆ? ความรุนแรงของอาการเป็นอย่างไรบ้าง?"
}
```

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

- **Attribution**: You must give appropriate credit
- **NonCommercial**: For research and non-commercial use only
- **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license

## 👥 Contributors

- zombitx64 : <zombitx64@gmail.com>

## 📚 References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [DeepSeek AI](https://deepseek.ai/)
- [LoRA/PEFT](https://github.com/huggingface/peft)
- [DeepSpeed](https://www.deepspeed.ai/)

## 📚 Citation

```bibtex
@misc{wang2024xmodellm1.5,
    title={Xmodel-LM1.5: An 1B-scale Multilingual LLM},
    author={Qun Wang and Yang Liu and QingQuan Lin  and Ling Jiang},
    year={2024},
    eprint={2411.10083},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

---
[Read the Xmodel-LM1.5 paper on arXiv](https://arxiv.org/pdf/2411.10083)
> **MedMT** © 2025 | Released under CC BY-NC-SA 4.0 | For research purposes only
