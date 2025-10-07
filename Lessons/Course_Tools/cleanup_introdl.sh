#!/bin/bash
# Complete cleanup script for introdl package and Course_Tools directory
# This removes all traces of old installations, cached versions, and obsolete files

echo "🧹 Complete introdl cleanup script"
echo "=================================="

# Detect script location dynamically
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COURSE_TOOLS="$SCRIPT_DIR"
INTRODL_SRC="$COURSE_TOOLS/introdl"

# Detect Python version dynamically
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"
echo ""

# ===================================================================
# Step 1: Clean Course_Tools directory
# ===================================================================
echo "Step 1: Cleaning Course_Tools directory..."
cd "$COURSE_TOOLS"

# Remove tilde files (CoCalc backup files)
if ls *~ 1> /dev/null 2>&1; then
    rm -f *~ 2>/dev/null && echo "  ✓ Removed backup files (*~)"
fi

# Remove __pycache__
if [ -d "__pycache__" ]; then
    rm -rf __pycache__ 2>/dev/null && echo "  ✓ Removed __pycache__"
fi

# Remove obsolete files from old package structure
OBSOLETE_FILES=(
    "diagnose_hyperstack.py"
    "diagnose_introdl.py"
    "force_clean_introdl.py"
    "hyperstack_clean_introdl.py"
    "update_introdl.py"
    "update_introdl.sh"
    "setup_course.sh"
    "To_Do_List.ipynb"
)

removed_count=0
for file in "${OBSOLETE_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm -f "$file" 2>/dev/null
        removed_count=$((removed_count + 1))
    fi
done

if [ $removed_count -gt 0 ]; then
    echo "  ✓ Removed $removed_count obsolete file(s)"
fi

# ===================================================================
# Step 2: Uninstall via pip
# ===================================================================
echo ""
echo "Step 2: Uninstalling via pip..."
for i in {1..3}; do
    python3 -m pip uninstall introdl -y 2>/dev/null
done
echo "  ✓ Completed pip uninstall attempts"

# ===================================================================
# Step 3: Remove from all possible site-packages
# ===================================================================
echo ""
echo "Step 3: Removing from all site-packages locations..."
LOCATIONS=(
    "$HOME/.local/lib/python${PYTHON_VERSION}/site-packages"
    "/usr/local/lib/python${PYTHON_VERSION}/dist-packages"
    "/usr/lib/python3/dist-packages"
    "/usr/lib/python${PYTHON_VERSION}/dist-packages"
)

cleaned_locations=0
for loc in "${LOCATIONS[@]}"; do
    if [ -d "$loc/introdl" ]; then
        echo "  Found introdl at: $loc"
        rm -rf "$loc/introdl" "$loc/introdl"*.{egg,dist}-info 2>/dev/null
        echo "  ✓ Removed"
        cleaned_locations=$((cleaned_locations + 1))
    fi
done

if [ $cleaned_locations -eq 0 ]; then
    echo "  (no installations found in site-packages)"
else
    echo "  ✓ Cleaned $cleaned_locations location(s)"
fi

# ===================================================================
# Step 4: Clean source directory build artifacts
# ===================================================================
echo ""
echo "Step 4: Cleaning source directory..."
cd "$INTRODL_SRC"
rm -rf build dist src/introdl.egg-info introdl.egg-info 2>/dev/null
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "  ✓ Cleaned build artifacts"

# ===================================================================
# Step 5: Clear pip cache
# ===================================================================
echo ""
echo "Step 5: Clearing pip cache..."
python3 -m pip cache remove introdl 2>/dev/null && echo "  ✓ Removed introdl from pip cache" || echo "  (pip cache disabled or already clear)"

# ===================================================================
# Completion
# ===================================================================
echo ""
echo "=================================="
echo "✅ Cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. If running in Jupyter: Restart your kernel"
echo "  2. Run the auto-update cell in your notebook"
echo "  3. Alternatively, install manually:"
echo "     cd $INTRODL_SRC"
echo "     python3 -m pip install . --no-cache-dir --force-reinstall"
echo ""
