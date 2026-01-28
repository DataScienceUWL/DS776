#!/bin/bash
# update_course_tools.sh - Update only Course_Tools from the DS776 GitHub repo
#
# Usage: bash ~/Lessons/Course_Tools/update_course_tools.sh
#
# Downloads the latest Lessons/Course_Tools from GitHub and overwrites
# the local copy. Does not touch any other files (Homework, Lessons, etc.)

set -e

REPO_URL="https://github.com/DataScienceUWL/DS776/archive/refs/heads/main.tar.gz"
REPO_NAME="DS776-main"
TEMP_DIR=$(mktemp -d)

# Determine target: use ~/Lessons/Course_Tools if it exists, otherwise ./Lessons/Course_Tools
if [[ -d "$HOME/Lessons/Course_Tools" ]]; then
    TARGET_DIR="$HOME/Lessons/Course_Tools"
elif [[ -d "./Lessons/Course_Tools" ]]; then
    TARGET_DIR="./Lessons/Course_Tools"
else
    echo "ERROR: Cannot find Lessons/Course_Tools directory"
    exit 1
fi

cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

echo "Downloading latest Course_Tools from GitHub..."
curl -sL "$REPO_URL" -o "$TEMP_DIR/repo.tar.gz"

echo "Extracting..."
tar -xzf "$TEMP_DIR/repo.tar.gz" -C "$TEMP_DIR"

SRC="$TEMP_DIR/$REPO_NAME/Lessons/Course_Tools"
if [[ ! -d "$SRC" ]]; then
    echo "ERROR: Course_Tools not found in downloaded archive"
    exit 1
fi

echo "Updating $TARGET_DIR ..."
rsync -a --delete "$SRC/" "$TARGET_DIR/"

echo "Done. Course_Tools updated to latest version."
