#!/bin/bash

# DS776 Course Setup Script
# This script sets up the course environment for all student accounts
# Run as: bash setup_course.sh

echo "=========================================="
echo "DS776 Deep Learning Course Setup"
echo "=========================================="
echo ""

# Determine the course root directory
if [ -n "$DS776_ROOT_DIR" ]; then
    COURSE_ROOT="$DS776_ROOT_DIR"
    echo "✅ Using DS776_ROOT_DIR: $COURSE_ROOT"
elif [ -d "$HOME/Lessons" ] && [ -d "$HOME/Homework" ]; then
    COURSE_ROOT="$HOME"
    echo "✅ Found course structure at: $COURSE_ROOT"
elif [ -d "./Lessons" ] && [ -d "./Homework" ]; then
    COURSE_ROOT="$(pwd)"
    echo "✅ Found course structure at: $COURSE_ROOT"
else
    COURSE_ROOT="$HOME/DS776"
    echo "⚠️  Course structure not found, will create at: $COURSE_ROOT"
fi

echo ""
echo "📦 Step 1: Installing introdl package..."
echo "=========================================="

# Find the introdl package
INTRODL_DIR="$COURSE_ROOT/Lessons/Course_Tools/introdl"

if [ -d "$INTRODL_DIR" ]; then
    echo "Found introdl package at: $INTRODL_DIR"
    echo "Installing package (standard mode for CoCalc compatibility)..."
    pip install "$INTRODL_DIR" --quiet
    if [ $? -eq 0 ]; then
        echo "✅ Package installed successfully!"
        echo "   Note: Using standard install (not editable) for CoCalc compatibility"
    else
        echo "❌ Package installation failed"
        exit 1
    fi
else
    echo "❌ Could not find introdl package at: $INTRODL_DIR"
    echo "   Please ensure the course files are properly installed"
    exit 1
fi

echo ""
echo "🔑 Step 2: Setting up API keys configuration..."
echo "=========================================="

# Note: config_paths_keys will create workspace directories automatically
# We just need to ensure api_keys.env exists in home_workspace
HOME_WORKSPACE="$COURSE_ROOT/home_workspace"
mkdir -p "$HOME_WORKSPACE"  # Ensure directory exists for api_keys

# Copy api_keys.env template if it doesn't exist
API_KEYS_DEST="$HOME_WORKSPACE/api_keys.env"
API_KEYS_TEMPLATE="$COURSE_ROOT/Lessons/Course_Tools/api_keys.env"

if [ ! -f "$API_KEYS_DEST" ]; then
    if [ -f "$API_KEYS_TEMPLATE" ]; then
        cp "$API_KEYS_TEMPLATE" "$API_KEYS_DEST"
        echo "✅ Created API keys file: $API_KEYS_DEST"
        echo ""
        echo "📝 TO ADD API KEYS:"
        echo "   1. Edit $API_KEYS_DEST"
        echo "   2. Replace placeholder values with actual API keys"
        echo "   3. Keys will automatically sync across compute servers"
    else
        echo "⚠️  API keys template not found at: $API_KEYS_TEMPLATE"
    fi
else
    echo "✅ API keys file already exists: $API_KEYS_DEST"
    echo "   (Existing keys have been preserved)"
fi

echo ""
echo "🔍 Step 3: Verifying installation..."
echo "=========================================="

# Test Python import
python3 -c "import introdl; print(f'✅ introdl version {introdl.__version__} installed successfully')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Could not verify introdl installation"
    echo "   You may need to restart your Python kernel"
else
    # Test config_paths_keys (this will create workspace directories)
    echo ""
    echo "Testing configuration and creating workspace directories..."
    python3 -c "
import os
os.environ['DS776_ROOT_DIR'] = '$COURSE_ROOT'
from introdl.utils import config_paths_keys
paths = config_paths_keys()
print('✅ Workspace directories created automatically')
print(f'   Data:   {paths[\"DATA_PATH\"]}')
print(f'   Models: {paths[\"MODELS_PATH\"]}')
print(f'   Cache:  {paths[\"CACHE_PATH\"]}')
" 2>/dev/null
    
    # Test PyTorch
    python3 -c "import torch; print(f'✅ PyTorch version {torch.__version__} available')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "⚠️  PyTorch not installed or not available"
    fi
fi

echo ""
echo "=========================================="
echo "🎉 Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add your API keys to: $API_KEYS_DEST"
echo "2. Start with Lesson 01 notebooks"
echo "3. Workspace folders are created automatically when you run notebooks"
echo ""
echo "For help:"
echo "- Storage issues: Use Clean_and_Free_Space.ipynb"
echo "- Package updates: Run this script again"
echo "- API key issues: Check $API_KEYS_DEST"
echo ""
echo "Note: The introdl package automatically creates workspace directories"
echo "      (home_workspace, cs_workspace, and Lesson/Homework Models folders)"
echo "      when you import and use config_paths_keys() in your notebooks."
echo ""