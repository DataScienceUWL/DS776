#!/bin/bash

# Script to sync course content (Lessons and Homework folders)
# Usage: 
#   ./sync_course_content.sh          # Default: backup and pull changes
#   ./sync_course_content.sh --refresh # Backup and completely refresh from repo

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository. Please run this script from the DS776 directory."
    exit 1
fi

# Get the repository root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Check if we're in the DS776 repository
if [[ ! -d "Lessons" ]] || [[ ! -d "Homework" ]]; then
    print_error "Lessons or Homework directories not found. Are you in the DS776 repository?"
    exit 1
fi

# Parse command line arguments
REFRESH_MODE=false
if [[ "$1" == "--refresh" ]] || [[ "$1" == "-r" ]]; then
    REFRESH_MODE=true
    print_info "Running in REFRESH mode - will completely reset Lessons and Homework from repo"
else
    print_info "Running in UPDATE mode - will pull latest changes while preserving local work"
fi

# Create backup directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${REPO_ROOT}/Developer/Backups/backup_${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

print_info "Creating backup in: ${BACKUP_DIR}"

# Function to backup a directory
backup_directory() {
    local dir_name=$1
    if [[ -d "$dir_name" ]]; then
        print_info "Backing up ${dir_name}..."
        cp -r "$dir_name" "${BACKUP_DIR}/${dir_name}"
        print_success "${dir_name} backed up"
    else
        print_warning "${dir_name} not found, skipping backup"
    fi
}

# Backup Lessons and Homework
backup_directory "Lessons"
backup_directory "Homework"

# Save current git status to backup
print_info "Saving git status to backup..."
git status > "${BACKUP_DIR}/git_status.txt"
git diff > "${BACKUP_DIR}/git_diff.txt"
git diff --staged > "${BACKUP_DIR}/git_diff_staged.txt"

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: ${CURRENT_BRANCH}" > "${BACKUP_DIR}/git_info.txt"
echo "Last commit: $(git log -1 --oneline)" >> "${BACKUP_DIR}/git_info.txt"

print_success "Backup completed in ${BACKUP_DIR}"

# Now perform the sync operation
if [[ "$REFRESH_MODE" == true ]]; then
    # REFRESH MODE: Complete reset from repository
    print_warning "Refreshing Lessons and Homework from repository..."
    
    # Stash any uncommitted changes
    print_info "Stashing uncommitted changes..."
    git stash push -m "sync_course_content: Auto-stash before refresh ${TIMESTAMP}"
    
    # Remove local Lessons and Homework directories
    print_info "Removing local Lessons and Homework directories..."
    rm -rf Lessons Homework
    
    # Checkout from repository
    print_info "Restoring clean versions from repository..."
    git checkout HEAD -- Lessons Homework
    
    print_success "Lessons and Homework refreshed from repository!"
    print_info "Your previous versions are backed up in: ${BACKUP_DIR}"
    print_info "Any uncommitted changes were stashed. Use 'git stash list' to see them."
    
else
    # UPDATE MODE: Pull latest changes
    print_info "Pulling latest changes from repository..."
    
    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        print_warning "You have uncommitted changes. Stashing them temporarily..."
        git stash push -m "sync_course_content: Auto-stash before pull ${TIMESTAMP}"
        STASHED=true
    else
        STASHED=false
    fi
    
    # Pull latest changes
    if git pull origin "${CURRENT_BRANCH}"; then
        print_success "Successfully pulled latest changes"
    else
        print_error "Failed to pull changes. Check for merge conflicts."
        if [[ "$STASHED" == true ]]; then
            print_info "Your changes are still stashed. Use 'git stash pop' to restore them."
        fi
        exit 1
    fi
    
    # Restore stashed changes if any
    if [[ "$STASHED" == true ]]; then
        print_info "Restoring your uncommitted changes..."
        if git stash pop; then
            print_success "Your changes have been restored"
        else
            print_warning "Could not automatically restore changes due to conflicts"
            print_info "Use 'git stash list' and 'git stash pop' to manually restore"
        fi
    fi
fi

# Final summary
echo ""
print_success "Sync operation completed!"
echo -e "${BLUE}Summary:${NC}"
echo "  • Backup location: ${BACKUP_DIR}"
echo "  • Current branch: ${CURRENT_BRANCH}"
if [[ "$REFRESH_MODE" == true ]]; then
    echo "  • Operation: Full refresh from repository"
else
    echo "  • Operation: Updated with latest changes"
fi

# Show current git status
echo ""
print_info "Current git status:"
git status --short

# Reminder about backup
echo ""
print_info "💡 Tip: To restore from backup, use:"
echo "  cp -r ${BACKUP_DIR}/Lessons ."
echo "  cp -r ${BACKUP_DIR}/Homework ."