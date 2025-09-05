#!/bin/bash

# DS776 Auto-Update introdl Script
# This script checks if introdl needs updating and does minimal setup
# Run from any lesson or homework notebook - it's fast and smart!

# Determine the course root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COURSE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INTRODL_DIR="$SCRIPT_DIR/introdl"

# Colors for output (if terminal supports them)
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Quick function to print with color
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

echo "🔍 DS776 introdl Auto-Check"
echo "=========================="

# Step 1: Check if introdl source exists
if [ ! -d "$INTRODL_DIR" ]; then
    print_error "introdl source not found at $INTRODL_DIR"
    echo ""
    print_info "This usually means you don't have the complete course files."
    print_info "Solutions:"
    print_info "1. Download the complete DS776 course repository from GitHub"
    print_info "2. Make sure you have the Lessons/Course_Tools/introdl directory"
    print_info "3. Set DS776_ROOT_DIR to your course directory if using local setup"
    echo ""
    print_info "If you're in CoCalc, contact the instructor - the files may not be synced."
    exit 1
fi

# Step 2: Get source version from __init__.py
SOURCE_VERSION=""
if [ -f "$INTRODL_DIR/src/introdl/__init__.py" ]; then
    SOURCE_VERSION=$(grep -oP "__version__\s*=\s*['\"]\\K[^'\"]*" "$INTRODL_DIR/src/introdl/__init__.py" 2>/dev/null)
fi

if [ -z "$SOURCE_VERSION" ]; then
    print_warning "Could not determine source version, will assume update needed"
    SOURCE_VERSION="unknown"
fi

echo "📦 Source version: $SOURCE_VERSION"

# Step 3: Check installed version
INSTALLED_VERSION=""
INSTALL_LOCATION=""
if python3 -c "import introdl" 2>/dev/null; then
    INSTALLED_VERSION=$(python3 -c "import introdl; print(getattr(introdl, '__version__', 'unknown'))" 2>/dev/null)
    INSTALL_LOCATION=$(python3 -c "import introdl; print(introdl.__file__)" 2>/dev/null)
    print_status "Installed version: $INSTALLED_VERSION"
    print_info "Location: $INSTALL_LOCATION"
else
    print_warning "introdl not installed"
fi

# Step 4: Version comparison logic
NEEDS_UPDATE=false

if [ -z "$INSTALLED_VERSION" ]; then
    print_info "No installation found - will install"
    NEEDS_UPDATE=true
elif [ "$INSTALLED_VERSION" = "unknown" ] || [ "$SOURCE_VERSION" = "unknown" ]; then
    print_info "Version comparison uncertain - will update to be safe"
    NEEDS_UPDATE=true
elif [ "$INSTALLED_VERSION" != "$SOURCE_VERSION" ]; then
    print_info "Version mismatch ($INSTALLED_VERSION != $SOURCE_VERSION) - will update"
    NEEDS_UPDATE=true
else
    print_status "Version $INSTALLED_VERSION is current - no update needed"
fi

# Step 5: Check if using local source (path conflict issue)
if [[ "$INSTALL_LOCATION" == *"Course_Tools"* ]]; then
    print_warning "Detected path conflict - using local source instead of installed package"
    NEEDS_UPDATE=true
fi

# Step 6: Install/update if needed
if [ "$NEEDS_UPDATE" = true ]; then
    echo ""
    echo "📦 Installing/updating introdl..."
    echo "================================"
    
    # Uninstall existing
    pip uninstall introdl -y --quiet 2>/dev/null
    
    # Install fresh
    if pip install "$INTRODL_DIR" --quiet 2>/dev/null; then
        print_status "Installation successful!"
        
        # Test the installation
        if python3 -c "from introdl.utils import config_paths_keys; print('✅ introdl.utils works!')" 2>/dev/null; then
            print_status "Installation verified - introdl.utils imports correctly"
        else
            print_warning "Installation may need kernel restart to work properly"
        fi
        
        echo ""
        echo "🔄 IMPORTANT: Restart your kernel and run this cell again!"
        echo "========================================================="
        exit 2  # Special exit code to indicate restart needed
        
    else
        print_error "Installation failed"
        echo "📝 Fallback: Try running the full Course_Setup.ipynb notebook"
        exit 1
    fi
fi

# Step 7: Quick health checks (only if no update was needed)
echo ""
echo "🔧 Quick health check..."

# Check introdl.utils import
if python3 -c "from introdl.utils import config_paths_keys" 2>/dev/null; then
    print_status "introdl.utils imports correctly"
else
    print_error "introdl.utils import failed - may need kernel restart"
fi

# Check workspace setup
HOME_WORKSPACE="$COURSE_ROOT/home_workspace"
if [ ! -d "$HOME_WORKSPACE" ]; then
    mkdir -p "$HOME_WORKSPACE"
    print_info "Created home_workspace directory"
fi

# Check API keys file
API_KEYS_FILE="$HOME_WORKSPACE/api_keys.env"
if [ ! -f "$API_KEYS_FILE" ]; then
    if [ -f "$SCRIPT_DIR/api_keys.env" ]; then
        cp "$SCRIPT_DIR/api_keys.env" "$API_KEYS_FILE"
        print_info "Created API keys template at home_workspace/api_keys.env"
        echo "📝 Remember to add your actual API keys for Lessons 7+"
    fi
else
    print_status "API keys file exists"
fi

# Check if we're in a local git repo (vs CoCalc/cloud environment)
IS_LOCAL_REPO=false
if [ -d "$COURSE_ROOT/.git" ]; then
    IS_LOCAL_REPO=true
    print_info "Detected local git repository"
    
    # Check for newer version on GitHub (only if network available)
    if command -v curl >/dev/null 2>&1 && command -v git >/dev/null 2>&1; then
        # Get the GitHub repo URL from git remote
        GITHUB_URL=$(git -C "$COURSE_ROOT" remote get-url origin 2>/dev/null)
        if [[ "$GITHUB_URL" == *"github.com"* ]]; then
            # Extract repo path and construct raw URL
            REPO_PATH=$(echo "$GITHUB_URL" | sed -E 's/.*github\.com[\/:]([^\/]+\/[^\/]+)(\.git)?$/\1/' | sed 's/\.git$//')
            GITHUB_VERSION=$(curl -s --connect-timeout 5 "https://raw.githubusercontent.com/$REPO_PATH/main/Lessons/Course_Tools/introdl/src/introdl/__init__.py" 2>/dev/null | grep -oP "__version__\s*=\s*['\"]\\K[^'\"]*" 2>/dev/null)
            
            if [ -n "$GITHUB_VERSION" ] && [ "$GITHUB_VERSION" != "$SOURCE_VERSION" ]; then
                echo ""
                print_warning "Newer version ($GITHUB_VERSION) available on GitHub!"
                print_info "To update: git pull origin main"
                print_info "Your current version: $SOURCE_VERSION"
            elif [ -n "$GITHUB_VERSION" ] && [ "$GITHUB_VERSION" = "$SOURCE_VERSION" ]; then
                print_status "You have the latest version from GitHub"
            fi
        fi
    fi
    
    # Check if local repo is behind remote
    if command -v git >/dev/null 2>&1; then
        # Fetch latest info (quietly)
        git -C "$COURSE_ROOT" fetch origin 2>/dev/null || true
        
        # Check if behind
        LOCAL_HASH=$(git -C "$COURSE_ROOT" rev-parse HEAD 2>/dev/null)
        REMOTE_HASH=$(git -C "$COURSE_ROOT" rev-parse origin/main 2>/dev/null)
        
        if [ -n "$LOCAL_HASH" ] && [ -n "$REMOTE_HASH" ] && [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
            echo ""
            print_warning "Your local repository is behind the remote!"
            print_info "To get latest course materials: git pull origin main"
            
            # Count commits behind
            COMMITS_BEHIND=$(git -C "$COURSE_ROOT" rev-list --count HEAD..origin/main 2>/dev/null)
            if [ -n "$COMMITS_BEHIND" ] && [ "$COMMITS_BEHIND" -gt 0 ]; then
                print_info "You are $COMMITS_BEHIND commits behind"
            fi
        fi
    fi
else
    print_info "Cloud environment (CoCalc/Colab) detected"
fi

echo ""
print_status "All checks complete - ready to proceed!"
echo "======================================"