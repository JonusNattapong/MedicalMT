#!/bin/bash
# Run the comprehensive XiaoduoAILab/XmodelLM1.5 model test

echo "==============================================="
echo " MedMT - XiaoduoAILab/XmodelLM1.5 Model Test"
echo "==============================================="
echo "Date: $(date)"
echo ""

# Ensure logs directory exists
mkdir -p logs

# Run the test
echo "Starting comprehensive model test..."
python tests/test_xmodel.py

# Find the latest test report
LATEST_REPORT=$(ls -t logs/xmodel_test_report_*.md | head -1)

if [ -n "$LATEST_REPORT" ]; then
    echo ""
    echo "Test completed. Report available at: $LATEST_REPORT"
    
    # Display a summary of the report
    echo ""
    echo "=== Test Report Summary ==="
    grep -A 2 "## Overall Status" "$LATEST_REPORT"
    grep -A 10 "## Test Results" "$LATEST_REPORT" | grep -v "## System Information" | head -7
    echo "=========================="
else
    echo ""
    echo "Test completed, but no report was generated."
fi

echo ""
echo "To view the full test logs, check the logs directory."
echo "==============================================="
