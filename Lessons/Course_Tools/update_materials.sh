#!/bin/bash
# update_materials.sh - Download/update DS776 course materials from GitHub
#
# Usage: bash update_materials.sh [--force]
#
# This script downloads Homework and Lessons folders from the DS776 GitHub repo.
# - Lessons: Full sync (overwrites all files - students don't edit these)
# - Homework: Safe sync (only adds NEW files, never overwrites existing files)
#
# Use --force to overwrite ALL files including existing Homework files
# (WARNING: This will overwrite any student work!)

set -e  # Exit on error

REPO_URL="https://github.com/DataScienceUWL/DS776/archive/refs/heads/main.tar.gz"
REPO_NAME="DS776-main"
TEMP_DIR=$(mktemp -d)

# Parse arguments
FORCE_MODE=false
if [[ "$1" == "--force" ]]; then
    FORCE_MODE=true
    echo "WARNING: Force mode enabled - ALL files will be overwritten!"
    echo ""
fi

echo "=========================================="
echo "DS776 Course Materials Update"
echo "=========================================="
echo ""

# Determine target directory (where Homework and Lessons should go)
# Default to home directory, but check if we're already in a course folder
if [[ -d "./Lessons" ]] || [[ -d "./Homework" ]]; then
    TARGET_DIR="."
    echo "Target: Current directory ($(pwd))"
else
    TARGET_DIR="$HOME"
    echo "Target: Home directory ($HOME)"
fi
echo ""

# Cleanup function
cleanup() {
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Download the repository
echo "Downloading course materials from GitHub..."
curl -sL "$REPO_URL" -o "$TEMP_DIR/repo.tar.gz"

if [[ ! -f "$TEMP_DIR/repo.tar.gz" ]]; then
    echo "ERROR: Failed to download repository"
    exit 1
fi

# Extract the archive
echo "Extracting archive..."
tar -xzf "$TEMP_DIR/repo.tar.gz" -C "$TEMP_DIR"

if [[ ! -d "$TEMP_DIR/$REPO_NAME" ]]; then
    echo "ERROR: Failed to extract repository"
    exit 1
fi

# Sync Lessons folder (full overwrite)
echo ""
echo "Updating Lessons folder (full sync)..."
if [[ -d "$TARGET_DIR/Lessons" ]]; then
    rsync -a --delete "$TEMP_DIR/$REPO_NAME/Lessons/" "$TARGET_DIR/Lessons/"
    echo "  Lessons folder updated."
else
    cp -r "$TEMP_DIR/$REPO_NAME/Lessons" "$TARGET_DIR/"
    echo "  Lessons folder created."
fi

# Sync Homework folder
echo ""
if [[ "$FORCE_MODE" == true ]]; then
    echo "Updating Homework folder (FORCE mode - overwriting all files)..."
    if [[ -d "$TARGET_DIR/Homework" ]]; then
        rsync -a --delete "$TEMP_DIR/$REPO_NAME/Homework/" "$TARGET_DIR/Homework/"
        echo "  Homework folder updated (all files overwritten)."
    else
        cp -r "$TEMP_DIR/$REPO_NAME/Homework" "$TARGET_DIR/"
        echo "  Homework folder created."
    fi
else
    echo "Updating Homework folder (safe mode - only adding new files)..."
    if [[ -d "$TARGET_DIR/Homework" ]]; then
        # Use rsync with --ignore-existing to only add new files
        rsync -a --ignore-existing "$TEMP_DIR/$REPO_NAME/Homework/" "$TARGET_DIR/Homework/"
        echo "  Homework folder updated (existing files preserved)."
    else
        cp -r "$TEMP_DIR/$REPO_NAME/Homework" "$TARGET_DIR/"
        echo "  Homework folder created."
    fi
fi

echo ""
echo "=========================================="
echo "Update complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Lessons: All files synced from GitHub"
if [[ "$FORCE_MODE" == true ]]; then
    echo "  - Homework: All files synced from GitHub (FORCE mode)"
else
    echo "  - Homework: New files added (your existing work preserved)"
fi
echo ""
echo "If you need to get fresh copies of homework files, run:"
echo "  bash ~/Lessons/Course_Tools/update_materials.sh --force"
echo ""
