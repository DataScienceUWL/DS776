#!/bin/bash

# Update introdl package - removes old package structure and installs the newest version
#
# This script:
# 1. Removes old subdirectory structure (idlmam/, utils/, visul/, nlp/) from introdl/src/introdl/
# 2. Removes all build artifacts (build/, dist/, *.egg-info, __pycache__)
# 3. Uninstalls any existing introdl package
# 4. Installs the fresh version from Lessons/Course_Tools/introdl
#
# Usage:
#     bash update_introdl.sh
#     or
#     ./update_introdl.sh

echo "============================================================"
echo "introdl Package Update Script"
echo "============================================================"

# Get the script directory (Course_Tools)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTRODL_DIR="${SCRIPT_DIR}/introdl"

if [ ! -d "$INTRODL_DIR" ]; then
    echo "Error: introdl package not found at $INTRODL_DIR"
    exit 1
fi

echo ""
echo "Working with introdl package at: $INTRODL_DIR"

# Step 1: Clean old subdirectory structure
echo ""
echo "1. Removing old package subdirectories..."

OLD_SUBDIRS=("idlmam" "utils" "visul" "nlp")
SRC_INTRODL="${INTRODL_DIR}/src/introdl"

# Check in src/introdl directory
if [ -d "$SRC_INTRODL" ]; then
    for subdir in "${OLD_SUBDIRS[@]}"; do
        OLD_PATH="${SRC_INTRODL}/${subdir}"
        if [ -d "$OLD_PATH" ]; then
            echo "   Removing old directory: $OLD_PATH"
            rm -rf "$OLD_PATH"
        fi
    done
fi

# Also check in the root introdl directory
for subdir in "${OLD_SUBDIRS[@]}"; do
    OLD_PATH="${INTRODL_DIR}/${subdir}"
    if [ -d "$OLD_PATH" ]; then
        echo "   Removing old directory: $OLD_PATH"
        rm -rf "$OLD_PATH"
    fi
done

# Step 2: Clean build artifacts
echo ""
echo "2. Cleaning build artifacts..."

# Remove build directories
[ -d "${INTRODL_DIR}/build" ] && echo "   Removing: ${INTRODL_DIR}/build" && rm -rf "${INTRODL_DIR}/build"
[ -d "${INTRODL_DIR}/dist" ] && echo "   Removing: ${INTRODL_DIR}/dist" && rm -rf "${INTRODL_DIR}/dist"

# Remove egg-info directories
find "$INTRODL_DIR" -type d -name "*.egg-info" -exec echo "   Removing: {}" \; -exec rm -rf {} + 2>/dev/null

# Step 3: Clean __pycache__ directories
echo ""
echo "3. Cleaning __pycache__ directories..."
find "$INTRODL_DIR" -type d -name "__pycache__" -exec echo "   Removing: {}" \; -exec rm -rf {} + 2>/dev/null

# Step 4: Remove .pyc and .pyo files
echo ""
echo "4. Cleaning compiled Python files..."
find "$INTRODL_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -exec echo "   Removing: {}" \; -exec rm -f {} +

# Step 4.5: Clean old introdl from site-packages
echo ""
echo "4.5. Cleaning old introdl from site-packages..."

# Get Python site-packages directory
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)

if [ -n "$SITE_PACKAGES" ] && [ -d "$SITE_PACKAGES/introdl" ]; then
    OLD_SUBDIRS=("idlmam" "utils" "visul" "nlp")
    for subdir in "${OLD_SUBDIRS[@]}"; do
        if [ -d "$SITE_PACKAGES/introdl/$subdir" ]; then
            echo "   Found old structure in site-packages: $SITE_PACKAGES/introdl/$subdir"
            echo "   Removing entire introdl directory: $SITE_PACKAGES/introdl"
            rm -rf "$SITE_PACKAGES/introdl"
            break
        fi
    done
fi

# Also check user site-packages
USER_SITE=$(python3 -c "import site; print(site.getusersitepackages())" 2>/dev/null)
if [ -n "$USER_SITE" ] && [ -d "$USER_SITE/introdl" ]; then
    OLD_SUBDIRS=("idlmam" "utils" "visul" "nlp")
    for subdir in "${OLD_SUBDIRS[@]}"; do
        if [ -d "$USER_SITE/introdl/$subdir" ]; then
            echo "   Found old structure in user site-packages: $USER_SITE/introdl/$subdir"
            echo "   Removing entire introdl directory: $USER_SITE/introdl"
            rm -rf "$USER_SITE/introdl"
            break
        fi
    done
fi

# Step 5: Uninstall existing introdl package
echo ""
echo "5. Uninstalling existing introdl package (if any)..."
pip uninstall -y introdl 2>/dev/null || true

# Step 6: Install fresh version
echo ""
echo "6. Installing fresh introdl package..."
cd "$INTRODL_DIR"

# Try pip install with editable mode first (for development)
if ! pip install -e . ; then
    echo "   Editable install failed, trying regular install..."
    if ! pip install . ; then
        echo ""
        echo "Error: Failed to install introdl package"
        exit 1
    fi
fi

# Step 7: Verify installation
echo ""
echo "7. Verifying installation..."

python3 -c "
import sys
try:
    import introdl
    print(f'✓ introdl successfully imported')
    print(f'  Version: {introdl.__version__}')

    # Test imports of main modules
    from introdl import utils, idlmam, visul, nlp
    print('✓ All main modules imported successfully')

    # Show installation location
    import os
    print(f'  Installed at: {os.path.dirname(introdl.__file__)}')

except ImportError as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "SUCCESS: introdl package updated successfully!"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "ERROR: Package installed but verification failed"
    echo "============================================================"
    exit 1
fi