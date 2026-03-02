#!/bin/bash
# Script to install additional dependencies for SHAP and LIME

echo "Installing SHAP and LIME for model explainability..."

# Try pip
python -m pip install shap lime --upgrade 2>/dev/null

# If that fails, try pip3
if [ $? -ne 0 ]; then
    python3 -m pip install shap lime --upgrade
fi
echo "Installation complete!"

