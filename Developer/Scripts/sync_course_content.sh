#!/bin/bash

# Script to sync course content (Lessons and Homework folders)
# Usage: 
#   ./sync_course_content.sh          # Default: backup and pull changes
#   ./sync_course_content.sh --refresh # Backup and completely refresh from repo
#   ./sync_course_content.sh --clone   # Clone fresh repo and refresh (CoCalc only)
#
# In CoCalc: Repository is maintained in ~/DS776_repo, working folders in ~
# In Local Dev: Everything is in the same git repository
#
# NOTE: Currently configured to work with feature/flexible-path-resolution branch
#       Update to 'main' branch after merge

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

# Parse command line arguments early to check for clone mode
CLONE_MODE=false
if [[ "$1" == "--clone" ]] || [[ "$1" == "-c" ]]; then
    CLONE_MODE=true
fi

# Determine the working directory based on environment
# Check for CoCalc environment indicators
IS_COCALC=false
if [[ -n "$COCALC_PROJECT_ID" ]] || [[ -f "$HOME/.smc" ]] || [[ -d "$HOME/.cocalc" ]]; then
    IS_COCALC=true
fi

if [[ "$IS_COCALC" == true ]] || ([[ "$HOME" == "/home/"* ]] && [[ "$1" == "--clone" ]]); then
    # CoCalc environment or likely CoCalc with --clone flag
    print_info "Detected CoCalc environment"
    WORK_DIR="$HOME"  # Where Lessons and Homework folders live
    REPO_DIR="$HOME/DS776_repo"  # Where the git repo is
    
    # Only verify repo exists if not in clone mode
    if [[ "$CLONE_MODE" != true ]]; then
        if [[ ! -d "$REPO_DIR/.git" ]]; then
            print_error "Repository not found at ~/DS776_repo"
            print_error "Please run with --clone flag to clone the repository first"
            exit 1
        fi
        if [[ ! -d "$REPO_DIR/Lessons" ]] || [[ ! -d "$REPO_DIR/Homework" ]]; then
            print_error "Repository incomplete at ~/DS776_repo"
            print_error "Please run with --clone flag to get a fresh copy"
            exit 1
        fi
    fi
elif git rev-parse --git-dir > /dev/null 2>&1; then
    # Local development: everything in the git repository
    REPO_ROOT=$(git rev-parse --show-toplevel)
    if [[ -d "$REPO_ROOT/Lessons" ]] && [[ -d "$REPO_ROOT/Homework" ]]; then
        print_info "Detected local development environment"
        WORK_DIR="$REPO_ROOT"  # Working folders are in the repo
        REPO_DIR="$REPO_ROOT"  # Repo is the same location
    else
        print_error "Not in the DS776 repository. Lessons or Homework directories not found."
        exit 1
    fi
else
    print_error "Could not detect environment. Please run from:"
    print_error "  - CoCalc: anywhere (use --clone if repo doesn't exist)"
    print_error "  - Local: from within the DS776 repository"
    exit 1
fi

# Change to work directory
cd "$WORK_DIR"

# Parse remaining command line arguments (CLONE_MODE already set above)
REFRESH_MODE=false
if [[ "$1" == "--refresh" ]] || [[ "$1" == "-r" ]]; then
    REFRESH_MODE=true
    print_info "Running in REFRESH mode - will completely reset Lessons and Homework from repo"
elif [[ "$CLONE_MODE" == true ]]; then
    print_info "Running in CLONE mode - will clone fresh repo and reset Lessons and Homework"
else
    print_info "Running in UPDATE mode - will pull latest changes while preserving local work"
fi

# Create backup directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# In CoCalc, create backups in home directory; in local dev, use repo
if [[ "$WORK_DIR" == "$HOME" ]]; then
    BACKUP_DIR="${HOME}/backups/backup_${TIMESTAMP}"
else
    BACKUP_DIR="${REPO_DIR}/Developer/Backups/backup_${TIMESTAMP}"
fi
mkdir -p "$BACKUP_DIR"

print_info "Creating backup in: ${BACKUP_DIR}"

# Function to backup a directory
backup_directory() {
    local dir_name=$1
    local source_path="${WORK_DIR}/${dir_name}"
    if [[ -d "$source_path" ]]; then
        print_info "Backing up ${dir_name}..."
        cp -r "$source_path" "${BACKUP_DIR}/${dir_name}"
        print_success "${dir_name} backed up"
    else
        print_warning "${dir_name} not found in ${WORK_DIR}, skipping backup"
    fi
}

# Backup Lessons and Homework from work directory
backup_directory "Lessons"
backup_directory "Homework"

# Save git status if in a repo (skip for CoCalc working directory)
if [[ "$WORK_DIR" != "$HOME" ]]; then
    print_info "Saving git status to backup..."
    git status > "${BACKUP_DIR}/git_status.txt"
    git diff > "${BACKUP_DIR}/git_diff.txt"
    git diff --staged > "${BACKUP_DIR}/git_diff_staged.txt"
    
    # Get current branch
    CURRENT_BRANCH=$(git branch --show-current)
    echo "Current branch: ${CURRENT_BRANCH}" > "${BACKUP_DIR}/git_info.txt"
    echo "Last commit: $(git log -1 --oneline)" >> "${BACKUP_DIR}/git_info.txt"
else
    # For CoCalc, get git info from the repo directory (if it exists)
    if [[ -d "$REPO_DIR/.git" ]]; then
        cd "$REPO_DIR"
        CURRENT_BRANCH=$(git branch --show-current)
        echo "Repository location: ${REPO_DIR}" > "${BACKUP_DIR}/git_info.txt"
        echo "Current branch: ${CURRENT_BRANCH}" >> "${BACKUP_DIR}/git_info.txt"
        echo "Last commit: $(git log -1 --oneline)" >> "${BACKUP_DIR}/git_info.txt"
        cd "$WORK_DIR"
    else
        # No repo yet (clone mode)
        CURRENT_BRANCH="main"  # Default branch
        echo "Repository will be cloned to: ${REPO_DIR}" > "${BACKUP_DIR}/git_info.txt"
    fi
fi

print_success "Backup completed in ${BACKUP_DIR}"

# Handle clone mode if requested
if [[ "$CLONE_MODE" == true ]]; then
    if [[ "$WORK_DIR" == "$HOME" ]]; then
        # CoCalc environment - clone fresh repo
        print_warning "Cloning fresh repository..."
        
        # Get the repo URL (try to detect from existing repo if possible)
        if [[ -d "$REPO_DIR/.git" ]]; then
            cd "$REPO_DIR"
            REPO_URL=$(git config --get remote.origin.url)
            cd "$WORK_DIR"
            
            # If GITHUB_PAT is set and URL doesn't have it, add it
            if [[ -n "$GITHUB_PAT" ]] && [[ "$REPO_URL" == "https://github.com/"* ]]; then
                # Extract the repo path and add PAT
                REPO_PATH=${REPO_URL#https://github.com/}
                REPO_URL="https://${GITHUB_PAT}@github.com/${REPO_PATH}"
                print_info "Using GITHUB_PAT for authentication"
            fi
        else
            # Default repo URL
            if [[ -n "$GITHUB_PAT" ]]; then
                REPO_URL="https://${GITHUB_PAT}@github.com/DataScienceUWL/DS776.git"
                print_info "Using default repo with GITHUB_PAT"
            else
                REPO_URL="https://github.com/DataScienceUWL/DS776.git"
                print_info "Using default repo URL"
            fi
        fi
        
        # Remove old repo
        print_info "Removing old repository..."
        rm -rf "$REPO_DIR"
        
        # Clone fresh repo
        print_info "Cloning from: ${REPO_URL//${GITHUB_PAT}@/[PAT]@}"  # Hide PAT in output
        if git clone "$REPO_URL" "$REPO_DIR"; then
            print_success "Repository cloned successfully"
            
            # Checkout the feature branch
            cd "$REPO_DIR"
            print_info "Checking out feature/flexible-path-resolution branch..."
            if git checkout feature/flexible-path-resolution; then
                print_success "Switched to feature/flexible-path-resolution branch"
            else
                print_warning "Could not checkout feature/flexible-path-resolution, staying on main"
            fi
            cd "$WORK_DIR"
            
            # Force refresh mode after clone
            REFRESH_MODE=true
        else
            print_error "Failed to clone repository"
            print_info "Your backup is at: ${BACKUP_DIR}"
            exit 1
        fi
    else
        print_error "Clone mode is only supported in CoCalc environment"
        print_info "In local development, use git commands directly"
        exit 1
    fi
fi

# Now perform the sync operation
if [[ "$WORK_DIR" == "$HOME" ]]; then
    # CoCalc mode: sync from repo to home directory
    
    # Update the repository (unless we just cloned it)
    if [[ "$CLONE_MODE" != true ]]; then
        print_info "Updating repository at ${REPO_DIR}..."
        cd "$REPO_DIR"
        
        # Pull latest changes in the repo
        # For now, use the feature branch
        if git pull origin feature/flexible-path-resolution; then
            print_success "Repository updated successfully"
        else
            # Fallback to current branch if feature branch doesn't exist
            print_warning "Could not pull feature branch, trying current branch..."
            if git pull origin "${CURRENT_BRANCH}"; then
                print_success "Repository updated successfully"
            else
                print_error "Failed to update repository. Check for issues in ${REPO_DIR}"
                exit 1
            fi
        fi
    fi
    
    # Now sync to home directory
    cd "$WORK_DIR"
    
    if [[ "$REFRESH_MODE" == true ]]; then
        # REFRESH MODE: Complete replacement from repository
        print_warning "Replacing Lessons and Homework with fresh copies from repository..."
        
        # Remove existing directories
        print_info "Removing existing Lessons and Homework directories..."
        rm -rf Lessons Homework
        
        # Copy from repository
        print_info "Copying fresh versions from repository..."
        cp -r "${REPO_DIR}/Lessons" .
        cp -r "${REPO_DIR}/Homework" .
        
        print_success "Lessons and Homework refreshed from repository!"
    else
        # UPDATE MODE: Sync changes while preserving local work
        print_info "Syncing changes from repository..."
        
        # Use rsync to intelligently sync, preserving local changes
        rsync -av --exclude='*.pyc' --exclude='__pycache__' --exclude='.ipynb_checkpoints' \
              "${REPO_DIR}/Lessons/" ./Lessons/
        rsync -av --exclude='*.pyc' --exclude='__pycache__' --exclude='.ipynb_checkpoints' \
              "${REPO_DIR}/Homework/" ./Homework/
        
        print_success "Successfully synced latest changes"
    fi
    
else
    # Local development mode: standard git operations
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
fi

# Final summary
echo ""
print_success "Sync operation completed!"
echo -e "${BLUE}Summary:${NC}"
echo "  • Backup location: ${BACKUP_DIR}"
echo "  • Current branch: ${CURRENT_BRANCH}"
if [[ "$CLONE_MODE" == true ]]; then
    echo "  • Operation: Cloned fresh repository and refreshed"
elif [[ "$REFRESH_MODE" == true ]]; then
    echo "  • Operation: Full refresh from repository"
else
    echo "  • Operation: Updated with latest changes"
fi

# Show current git status (only for local development)
if [[ "$WORK_DIR" != "$HOME" ]]; then
    echo ""
    print_info "Current git status:"
    git status --short
fi

# Reminder about backup
echo ""
print_info "💡 Tip: To restore from backup, use:"
echo "  cp -r ${BACKUP_DIR}/Lessons ."
echo "  cp -r ${BACKUP_DIR}/Homework ."