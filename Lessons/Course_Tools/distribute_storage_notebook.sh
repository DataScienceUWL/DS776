#!/bin/bash
#
# distribute_storage_notebook.sh
#
# Distributes Storage_Cleanup.ipynb from Lessons/Course_Tools/
# to all Homework folders, removing old Storage_Cleanup_After_HW.ipynb files.
#
# Usage:
#   bash Lessons/Course_Tools/distribute_storage_notebook.sh
#   OR (from Course_Tools directory):
#   bash distribute_storage_notebook.sh
#

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine DS776 root (should be two levels up from Course_Tools)
DS776_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source notebook path (new name)
SOURCE_NOTEBOOK="$SCRIPT_DIR/Storage_Cleanup.ipynb"

# Check if source notebook exists
if [ ! -f "$SOURCE_NOTEBOOK" ]; then
    echo -e "${YELLOW}⚠️  Source notebook not found: $SOURCE_NOTEBOOK${NC}"
    echo "   Make sure you're running this from the DS776 repository."
    exit 1
fi

echo "=========================================="
echo "📦 Storage Cleanup Notebook Distribution"
echo "=========================================="
echo ""
echo -e "${BLUE}Source:${NC} $SOURCE_NOTEBOOK"
echo -e "${BLUE}Root:${NC}   $DS776_ROOT"
echo ""

# Find all homework folders and distribute new notebook
HOMEWORK_DIR="$DS776_ROOT/Homework"
UPDATED_COUNT=0
FAILED_COUNT=0
REMOVED_COUNT=0

if [ ! -d "$HOMEWORK_DIR" ]; then
    echo -e "${YELLOW}⚠️  Homework directory not found: $HOMEWORK_DIR${NC}"
    exit 1
fi

# Iterate through homework folders
for hw_folder in "$HOMEWORK_DIR"/Homework_*/; do
    if [ -d "$hw_folder" ]; then
        HW_NAME=$(basename "$hw_folder")
        OLD_NOTEBOOK="${hw_folder}Storage_Cleanup_After_HW.ipynb"
        NEW_NOTEBOOK="${hw_folder}Storage_Cleanup.ipynb"

        # Remove old notebook if it exists
        if [ -f "$OLD_NOTEBOOK" ]; then
            if rm "$OLD_NOTEBOOK"; then
                echo -e "${BLUE}🗑️${NC}  Removed old: $HW_NAME/Storage_Cleanup_After_HW.ipynb"
                ((REMOVED_COUNT++))
            else
                echo -e "${YELLOW}⚠️${NC}  Failed to remove: $HW_NAME/Storage_Cleanup_After_HW.ipynb"
            fi
        fi

        # Copy the source notebook to a temp location first
        TEMP_NOTEBOOK="${NEW_NOTEBOOK}.tmp"
        if cp "$SOURCE_NOTEBOOK" "$TEMP_NOTEBOOK"; then
            # Fix the auto-update path for homework folders
            # Change: %run ./auto_update_introdl.py
            # To:     %run ../../Lessons/Course_Tools/auto_update_introdl.py
            sed -i 's|%run \./auto_update_introdl\.py|%run ../../Lessons/Course_Tools/auto_update_introdl.py|g' "$TEMP_NOTEBOOK"

            # Move temp file to final location
            if mv "$TEMP_NOTEBOOK" "$NEW_NOTEBOOK"; then
                echo -e "${GREEN}✅${NC} Updated: $HW_NAME/Storage_Cleanup.ipynb"
                ((UPDATED_COUNT++))
            else
                echo -e "${YELLOW}❌${NC} Failed:  $HW_NAME/Storage_Cleanup.ipynb"
                ((FAILED_COUNT++))
                rm -f "$TEMP_NOTEBOOK"  # Clean up temp file
            fi
        else
            echo -e "${YELLOW}❌${NC} Failed to copy: $HW_NAME/Storage_Cleanup.ipynb"
            ((FAILED_COUNT++))
        fi
    fi
done

# Summary
echo ""
echo "=========================================="
echo "📊 Distribution Summary"
echo "=========================================="
echo -e "${BLUE}🗑️  Removed:${NC} $REMOVED_COUNT old notebooks"
echo -e "${GREEN}✅ Updated:${NC} $UPDATED_COUNT new notebooks"

if [ $FAILED_COUNT -gt 0 ]; then
    echo -e "${YELLOW}❌ Failed:${NC}  $FAILED_COUNT notebooks"
fi

echo ""

if [ $UPDATED_COUNT -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No notebooks were updated.${NC}"
    echo "   Make sure homework folders exist in $HOMEWORK_DIR"
else
    echo -e "${BLUE}💡 Next steps:${NC}"
    echo "   1. Review changes with: git diff Homework/*/Storage_Cleanup.ipynb"
    echo "   2. Test a notebook: cd Homework/Homework_05 && jupyter notebook"
    echo "   3. Commit changes: git add . && git commit -m 'Rename storage notebooks to Storage_Cleanup.ipynb'"
fi

echo ""
