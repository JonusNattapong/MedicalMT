#!/bin/bash
# Complete MedMT workflow script for generating data, running models, and comparing results

# Set bash to exit on error
set -e

# Configuration
NUM_SAMPLES=50
OUTPUT_DIR="results"
DATASET_DIR="data"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATASET_NAME="medical_dialogues_${TIMESTAMP}.csv"
FULL_DATASET_PATH="${DATASET_DIR}/${DATASET_NAME}"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATASET_DIR}"

echo "===== MedMT Workflow ====="
echo "Date: $(date)"
echo "Generating ${NUM_SAMPLES} samples using DeepSeek Reasoner"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================="

# Step 1: Generate dataset with DeepSeek Reasoner
echo "[1/5] Generating dataset with DeepSeek Reasoner..."
python scripts/generate_full_dataset.py --samples "${NUM_SAMPLES}" --output-dir "${DATASET_DIR}" 
echo "Dataset generation completed."

# Get the most recent dataset file
LATEST_DATASET=$(ls -t "${DATASET_DIR}"/*.csv | head -1)
echo "Using dataset: ${LATEST_DATASET}"

# Step 2: Run XiaoduoAILab/XmodelLM1.5 translation
echo "[2/5] Running XiaoduoAILab/XmodelLM1.5 translations..."
python examples/evaluate_translations.py --dataset "${LATEST_DATASET}" --output "${OUTPUT_DIR}/xmodel_evaluation_${TIMESTAMP}.csv" --samples 10
echo "XmodelLM1.5 translation completed."

# Step 3: Run DeepSeek Reasoner translation
echo "[3/5] Running DeepSeek Reasoner translations..."
# We'll use a modified version of the evaluation script for DeepSeek
python examples/model_comparison.py
echo "DeepSeek Reasoner translation completed."

# Step 4: Compare models
echo "[4/5] Comparing translation results..."
python scripts/compare_metrics.py --results \
  "XmodelLM1.5:${OUTPUT_DIR}/xmodel_evaluation_${TIMESTAMP}.csv" \
  "DeepSeek:${OUTPUT_DIR}/deepseek_evaluation_${TIMESTAMP}.csv" \
  --output-dir "${OUTPUT_DIR}"
echo "Model comparison completed."

# Step 5: Generate final report
echo "[5/5] Generating final report..."
echo "# MedMT Translation Evaluation Report" > "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "Date: $(date)" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "## Dataset" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "- Source: ${LATEST_DATASET}" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "- Samples: ${NUM_SAMPLES}" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "## Models Evaluated" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "1. XiaoduoAILab/XmodelLM1.5" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "2. DeepSeek Reasoner (deepseek-ai/deepseek-coder-33b-instruct)" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "## Results" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "See metrics summary in \`metrics_summary_${TIMESTAMP}.csv\`" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "## Visualizations" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "![Metrics Comparison](./metrics_comparison_${TIMESTAMP}.png)" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "## Conclusion" >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "Add your conclusions here based on the results." >> "${OUTPUT_DIR}/report_${TIMESTAMP}.md"

echo "Final report generated: ${OUTPUT_DIR}/report_${TIMESTAMP}.md"
echo "=========================="
echo "MedMT workflow completed successfully!"
echo "Results available in: ${OUTPUT_DIR}"
