# MedMT Testing Suite

This directory contains tests for the MedMT project, focusing on verifying model functionality and evaluating translation quality.

## Available Tests

### XiaoduoAILab/XmodelLM1.5 Test

This comprehensive test validates the XiaoduoAILab/XmodelLM1.5 model functionality, checking:

- Model loading (both instruction-tuned and pretrained versions)
- Basic text generation
- Chinese to Thai translation
- Custom parameter configuration
- Custom system prompts

Run the test with:

```bash
python tests/test_xmodel.py
```

The test generates a detailed report in the `logs` directory with the format `xmodel_test_report_YYYYMMDD_HHMMSS.md`.

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a new Python file in the `tests` directory
2. Use appropriate logging to track test execution
3. Generate a test report with results
4. Document the test in this README

## Test Requirements

The testing suite requires these additional dependencies (included in the project's `requirements.txt`):

- pytest
- nltk (for BLEU/METEOR metrics)
- matplotlib/seaborn (for visualizations)

## Continuous Integration

Tests can be integrated into a CI/CD pipeline by:

1. Running test scripts in the pipeline
2. Validating that all tests pass before deployment
3. Generating test reports for each build

## Manual Testing

For manual testing of translation quality:

1. Generate a small test dataset with diverse medical content
2. Translate using both DeepSeek Reasoner and XmodelLM1.5
3. Compare translations using metrics and qualitative evaluation
4. Document any issues or improvements in the appropriate GitHub issue
