#!/bin/bash
# Setup script for nightly two-way sync between WSL and Google Drive
# This adds a cron job to run the sync script every night at 2 AM

SYNC_SCRIPT="$HOME/DS776/sync_bidirectional.sh"
CRON_TIME="0 2 * * *"  # 2 AM every day

echo "==================================="
echo "DS776 Nightly Sync Setup"
echo "==================================="
echo ""
echo "This will set up a cron job to run bidirectional sync"
echo "between WSL and Google Drive every night at 2 AM."
echo ""
echo "Script: $SYNC_SCRIPT"
echo "Schedule: $CRON_TIME (2 AM daily)"
echo ""

# Check if script exists
if [ ! -f "$SYNC_SCRIPT" ]; then
    echo "ERROR: Sync script not found at $SYNC_SCRIPT"
    exit 1
fi

# Check if script is executable
if [ ! -x "$SYNC_SCRIPT" ]; then
    echo "Making script executable..."
    chmod +x "$SYNC_SCRIPT"
fi

# Check if cron job already exists
CRON_ENTRY="$CRON_TIME $SYNC_SCRIPT >> $HOME/DS776/Developer/Notes/sync_logs/cron.log 2>&1"
if crontab -l 2>/dev/null | grep -q "$SYNC_SCRIPT"; then
    echo "⚠️  Cron job already exists for this script."
    echo ""
    echo "Current crontab:"
    crontab -l | grep "$SYNC_SCRIPT"
    echo ""
    read -p "Do you want to replace it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi

    # Remove old entry
    crontab -l | grep -v "$SYNC_SCRIPT" | crontab -
fi

# Add cron job
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo ""
echo "✅ Cron job added successfully!"
echo ""
echo "The sync will run every night at 2 AM."
echo ""
echo "Useful commands:"
echo "  View crontab:        crontab -l"
echo "  Remove cron job:     crontab -e  (then delete the line)"
echo "  View sync logs:      ls -lh ~/DS776/Developer/Notes/sync_logs/"
echo "  Test sync now:       bash $SYNC_SCRIPT"
echo ""
echo "NOTE: To change the time, edit this setup script and run again."
echo "      Or run: crontab -e"
echo ""
