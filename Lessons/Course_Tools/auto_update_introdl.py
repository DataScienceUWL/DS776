#!/usr/bin/env python3
"""
DS776 Auto-Update introdl Script (Python version for Windows compatibility)
This script checks if introdl needs updating and does minimal setup
Cross-platform: Works on Windows, Mac, Linux
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def print_colored(symbol, message, color_code=None):
    """Print with color if supported, plain text otherwise"""
    if color_code and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        print(f"\033[{color_code}m{symbol}\033[0m {message}")
    else:
        print(f"{symbol} {message}")

def print_status(message):
    print_colored("✅", message, "0;32")

def print_info(message):
    print_colored("ℹ️", message, "0;34")

def print_warning(message):
    print_colored("⚠️", message, "1;33")

def print_error(message):
    print_colored("❌", message, "0;31")

def main():
    print("🔍 DS776 introdl Auto-Check")
    print("=" * 26)
    
    # Determine the course root directory
    script_dir = Path(__file__).parent.absolute()
    course_root = script_dir.parent.parent
    introdl_dir = script_dir / "introdl"
    
    # Check if introdl source exists
    if not introdl_dir.exists():
        print_error(f"introdl source not found at {introdl_dir}")
        sys.exit(1)
    
    # Get source version from __init__.py
    source_version = None
    init_file = introdl_dir / "src" / "introdl" / "__init__.py"
    if init_file.exists():
        try:
            with open(init_file, 'r') as f:
                for line in f:
                    if '__version__' in line and '=' in line:
                        # Extract version string more carefully
                        import re
                        match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', line)
                        if match:
                            source_version = match.group(1)
                            break
        except Exception:
            pass
    
    if not source_version:
        print_warning("Could not determine source version, will assume update needed")
        source_version = "unknown"
    
    print(f"📦 Source version: {source_version}")
    
    # Check installed version
    installed_version = None
    install_location = None
    try:
        result = subprocess.run([
            sys.executable, "-c", "import introdl; print(introdl.__file__); print(getattr(introdl, '__version__', 'unknown'))"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                install_location = lines[0]
                installed_version = lines[1]
                print_status(f"Installed version: {installed_version}")
                print_info(f"Location: {install_location}")
    except Exception:
        pass
    
    if not installed_version:
        print_warning("introdl not installed")
    
    # Version comparison logic
    needs_update = False
    
    if not installed_version:
        print_info("No installation found - will install")
        needs_update = True
    elif installed_version == "unknown" or source_version == "unknown":
        print_info("Version comparison uncertain - will update to be safe")
        needs_update = True
    elif installed_version != source_version:
        print_info(f"Version mismatch ({installed_version} != {source_version}) - will update")
        needs_update = True
    else:
        print_status(f"Version {installed_version} is current - no update needed")
    
    # Check if using local source (path conflict issue)
    if install_location and "Course_Tools" in install_location:
        print_warning("Detected path conflict - using local source instead of installed package")
        needs_update = True
    
    # Install/update if needed
    if needs_update:
        print("\n📦 Installing/updating introdl...")
        print("=" * 32)
        
        # Uninstall existing
        try:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "introdl", "-y", "--quiet"], 
                         capture_output=True)
        except:
            pass
        
        # Install fresh
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", str(introdl_dir), "--quiet"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print_status("Installation successful!")
                
                # Test the installation
                test_result = subprocess.run([
                    sys.executable, "-c", "from introdl.utils import config_paths_keys; print('Works!')"
                ], capture_output=True, text=True)
                
                if test_result.returncode == 0:
                    print_status("Installation verified - introdl.utils imports correctly")
                else:
                    print_warning("Installation may need kernel restart to work properly")
                
                print("\n🔄 IMPORTANT: Restart your kernel and run this cell again!")
                print("=" * 57)
                sys.exit(2)  # Special exit code to indicate restart needed
                
            else:
                print_error("Installation failed")
                print("📝 Fallback: Try running the full Course_Setup.ipynb notebook")
                sys.exit(1)
                
        except Exception as e:
            print_error(f"Installation error: {e}")
            sys.exit(1)
    
    # Quick health checks (only if no update was needed)
    print("\n🔧 Quick health check...")
    
    # Check introdl.utils import
    try:
        subprocess.run([sys.executable, "-c", "from introdl.utils import config_paths_keys"], 
                      capture_output=True, text=True, check=True)
        print_status("introdl.utils imports correctly")
    except:
        print_error("introdl.utils import failed - may need kernel restart")
    
    # Check workspace setup
    home_workspace = course_root / "home_workspace"
    home_workspace.mkdir(parents=True, exist_ok=True)
    
    # Check API keys file
    api_keys_file = home_workspace / "api_keys.env"
    if not api_keys_file.exists():
        api_keys_template = script_dir / "api_keys.env"
        if api_keys_template.exists():
            import shutil
            shutil.copy2(api_keys_template, api_keys_file)
            print_info("Created API keys template at home_workspace/api_keys.env")
            print("📝 Remember to add your actual API keys for Lessons 7+")
        else:
            print_warning("API keys template not found")
    else:
        print_status("API keys file exists")
    
    # Check if we're in a local git repo
    if (course_root / ".git").exists():
        print_info("Detected local git repository")
        
        # Check for newer version on GitHub (if network available)
        try:
            import urllib.request
            import re
            
            # Get GitHub repo URL
            result = subprocess.run(
                ["git", "-C", str(course_root), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0 and "github.com" in result.stdout:
                github_url = result.stdout.strip()
                # Extract repo path
                match = re.search(r'github\.com[\/:]([^\/]+\/[^\/]+)(?:\.git)?$', github_url)
                if match:
                    repo_path = match.group(1).rstrip('.git')
                    raw_url = f"https://raw.githubusercontent.com/{repo_path}/main/Lessons/Course_Tools/introdl/src/introdl/__init__.py"
                    
                    # Fetch GitHub version
                    try:
                        with urllib.request.urlopen(raw_url, timeout=5) as response:
                            content = response.read().decode('utf-8')
                            version_match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content)
                            if version_match:
                                github_version = version_match.group(1)
                                
                                if github_version != source_version:
                                    print_warning(f"Newer version ({github_version}) available on GitHub!")
                                    print_info("To update: git pull origin main")
                                else:
                                    print_status("You have the latest version from GitHub")
                    except:
                        pass  # Network issues, skip silently
                        
        except:
            pass  # Git/network issues, skip silently
    else:
        print_info("Cloud environment (CoCalc/Colab) detected")
    
    print("")
    print_status("All checks complete - ready to proceed!")
    print("=" * 38)

if __name__ == "__main__":
    main()