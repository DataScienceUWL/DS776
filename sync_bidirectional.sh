#!/bin/bash
# DS776 Two-Way Sync Script - Bidirectional sync between WSL and Google Drive
# This script syncs changes in BOTH directions
# Usage: bash ~/DS776/sync_bidirectional.sh

set -e  # Exit on error

WSL_DIR="$HOME/DS776/"
GDRIVE_DIR="/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/"
LOG_DIR="$HOME/DS776/Developer/Notes/sync_logs"
LOG_FILE="$LOG_DIR/sync_$(date +%Y%m%d_%H%M%S).log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "==================================="
log "DS776 Two-Way Sync"
log "==================================="
log "WSL Directory:    $WSL_DIR"
log "Google Drive Dir: $GDRIVE_DIR"
log ""

# Check if Google Drive is mounted
if [ ! -d "$GDRIVE_DIR" ]; then
    log "ERROR: Google Drive directory not found!"
    log "Make sure your Google Drive is mounted at: $GDRIVE_DIR"
    exit 1
fi

log "Step 1: Syncing FROM Google Drive TO WSL (pulling remote changes)..."
log "---------------------------------------------------------------"

# Sync from Google Drive to WSL (pull remote changes)
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose
# -u: update (skip files that are newer on receiver)
# --exclude: skip certain directories/files
rsync -avu \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='Developer/Notes/sync_logs/' \
    "$GDRIVE_DIR" "$WSL_DIR" 2>&1 | tee -a "$LOG_FILE"

log ""
log "Step 2: Syncing FROM WSL TO Google Drive (pushing local changes)..."
log "---------------------------------------------------------------"

# Sync from WSL to Google Drive (push local changes)
rsync -avu \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='Developer/Notes/sync_logs/' \
    "$WSL_DIR" "$GDRIVE_DIR" 2>&1 | tee -a "$LOG_FILE"

log ""
log "==================================="
log "Two-way sync completed!"
log "==================================="

# Verify sizes
log ""
log "Verification:"
log "WSL size:         $(du -sh $WSL_DIR 2>/dev/null | cut -f1)"
log "Google Drive size: $(du -sh $GDRIVE_DIR 2>/dev/null | cut -f1)"

log ""
log "Log saved to: $LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "sync_*.log" -mtime +30 -delete 2>/dev/null || true

echo ""
echo "✅ Sync complete! Check log for details: $LOG_FILE"
