#!/bin/bash
# DS776 Backup Script - Sync from WSL to Google Drive
# Run this periodically to backup your work to Google Drive
# Usage: bash ~/DS776/backup_to_gdrive.sh

set -e  # Exit on error

SOURCE_DIR="$HOME/DS776/"
DEST_DIR="/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/"

echo "==================================="
echo "DS776 Backup to Google Drive"
echo "==================================="
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo ""
echo "Starting backup at $(date)"
echo ""

# Perform rsync with archive mode and progress
# -a: archive mode (preserves permissions, timestamps, etc.)
# --info=progress2: show overall progress
# --delete: remove files from destination that no longer exist in source
rsync -a --info=progress2 --delete "$SOURCE_DIR" "$DEST_DIR"

echo ""
echo "==================================="
echo "Backup completed at $(date)"
echo "==================================="

# Verify sizes match
echo ""
echo "Verification:"
echo "Source size:      $(du -sh $SOURCE_DIR | cut -f1)"
echo "Destination size: $(du -sh $DEST_DIR | cut -f1)"
