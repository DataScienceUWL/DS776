#!/bin/bash
# Script to clean Homework 11 and Lesson 11 directories for a fresh start
# Run this from the project root directory

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}  Clean Lesson 11 and Homework 11 Directories${NC}"
echo -e "${YELLOW}================================================${NC}"
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: Not in a git repository root directory${NC}"
    echo "Please run this script from the project root (where .git directory is)"
    exit 1
fi

# Define directories to clean
HOMEWORK_DIR="Homework/Homework_11"
LESSON_DIR="Lessons/Lesson_11_Text_Generation"

# Check if directories exist
if [ ! -d "$HOMEWORK_DIR" ]; then
    echo -e "${RED}Error: $HOMEWORK_DIR directory not found${NC}"
    exit 1
fi

if [ ! -d "$LESSON_DIR" ]; then
    echo -e "${RED}Error: $LESSON_DIR directory not found${NC}"
    exit 1
fi

# Show what will be deleted
echo -e "${YELLOW}Files to be removed from $HOMEWORK_DIR:${NC}"
ls -1 "$HOMEWORK_DIR"
echo ""

echo -e "${YELLOW}Files to be removed from $LESSON_DIR:${NC}"
ls -1 "$LESSON_DIR"
echo ""

# Ask for confirmation
read -p "Are you sure you want to remove all files from these directories? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${RED}Aborted by user${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}Removing files...${NC}"

# Remove files from Homework/Homework_11
if [ -d "$HOMEWORK_DIR" ]; then
    echo "Cleaning $HOMEWORK_DIR..."
    rm -rf "${HOMEWORK_DIR:?}"/*
    echo -e "${GREEN}✓ Homework_11 directory cleaned${NC}"
else
    echo -e "${RED}Warning: $HOMEWORK_DIR not found${NC}"
fi

# Remove files from Lessons/Lesson_11_Text_Generation
if [ -d "$LESSON_DIR" ]; then
    echo "Cleaning $LESSON_DIR..."
    rm -rf "${LESSON_DIR:?}"/*
    echo -e "${GREEN}✓ Lesson_11_Text_Generation directory cleaned${NC}"
else
    echo -e "${RED}Warning: $LESSON_DIR not found${NC}"
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Cleanup complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Directories are now empty and ready for new files."
echo "Use git status to see the changes."
echo ""
echo "To stage all deletions:"
echo "  git add -A"
echo ""
echo "To commit:"
echo "  git commit -m 'Clean Lesson 11 and Homework 11 for v2 deployment'"
echo ""
