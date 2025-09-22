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

# Global verbose flag
verbose = False

def print_colored(symbol, message, color_code=None):
    """Print with color if supported, plain text otherwise"""
    if color_code and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        print(f"\033[{color_code}m{symbol}\033[0m {message}")
    else:
        print(f"{symbol} {message}")

def print_status(message):
    print_colored("✅", message, "0;32")

def print_info(message, force=False):
    """Print info messages only in verbose mode, unless forced"""
    if verbose or force:
        print_colored("ℹ️", message, "0;34")

def print_warning(message):
    print_colored("⚠️", message, "1;33")

def print_error(message):
    print_colored("❌", message, "0;31")

def main():
    global verbose
    # Check for verbose flag
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    # Determine the course root directory
    script_dir = Path(__file__).parent.absolute()
    course_root = script_dir.parent.parent
    introdl_dir = script_dir / "introdl"

    # Check if introdl source exists
    if not introdl_dir.exists():
        print_error(f"introdl source not found at {introdl_dir}")
        print("")
        print_info("This usually means you don't have the complete course files.", force=True)
        print_info("Solutions:", force=True)
        print_info("1. Download the complete DS776 course repository from GitHub", force=True)
        print_info("2. Make sure you have the Lessons/Course_Tools/introdl directory", force=True)
        print_info("3. Set DS776_ROOT_DIR to your course directory if using local setup", force=True)
        print("")
        print_info("If you're in CoCalc, contact the instructor - the files may not be synced.", force=True)
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

    if verbose:
        print("📦 Source version: {}".format(source_version))

    # Check installed version with detailed diagnostics
    installed_version = None
    install_location = None

    # First check with pip (only in verbose mode)
    if verbose:
        pip_check = subprocess.run([
            sys.executable, "-m", "pip", "show", "introdl"
        ], capture_output=True, text=True)

        if pip_check.returncode == 0:
            # Parse pip show output
            pip_version = None
            pip_location = None
            for line in pip_check.stdout.split('\n'):
                if line.startswith('Version:'):
                    pip_version = line.split(':', 1)[1].strip()
                elif line.startswith('Location:'):
                    pip_location = line.split(':', 1)[1].strip()

            if pip_version:
                print_info(f"pip reports introdl version {pip_version} at {pip_location}")

    # Now try to import it
    try:
        result = subprocess.run([
            sys.executable, "-c", "import introdl; print(introdl.__file__); print(getattr(introdl, '__version__', 'unknown'))"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                install_location = lines[0]
                installed_version = lines[1]
                if verbose:
                    print_status(f"Installed version: {installed_version}")
                    print_info(f"Location: {install_location}")
        else:
            # Import failed, show why only in verbose mode
            if verbose and result.stderr:
                print_warning(f"Cannot import introdl: {result.stderr.strip()}")
    except Exception as e:
        if verbose:
            print_error(f"Error checking installation: {e}")

    if not installed_version and verbose:
        print_warning("introdl not installed or not importable")

    # Version comparison logic
    needs_update = False

    if not installed_version:
        if verbose:
            print_info("No installation found - will install")
        needs_update = True
    elif installed_version == "unknown" or source_version == "unknown":
        if verbose:
            print_info("Version comparison uncertain - will update to be safe")
        needs_update = True
    elif installed_version != source_version:
        if verbose:
            print_info(f"Version mismatch ({installed_version} != {source_version}) - will update")
        needs_update = True
    else:
        # Already up to date - only show this message
        print_status(f"introdl v{installed_version} already up to date")

    # Check if using local source (path conflict issue)
    if install_location and "Course_Tools" in install_location:
        if verbose:
            print_warning("Detected path conflict - using local source instead of installed package")
        needs_update = True

    # Install/update if needed
    if needs_update:
        if not verbose:
            # Simple message for non-verbose mode
            print("📦 Updating introdl to v{}...".format(source_version))
        else:
            print("\n📦 Installing/updating introdl...")
            print("=" * 32)

        # Uninstall existing
        if verbose:
            print_info("Uninstalling old version...")
        try:
            uninstall_result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "introdl", "-y"],
                         capture_output=True, text=True)
            if verbose:
                if "Successfully uninstalled" in uninstall_result.stdout:
                    print_status("Old version uninstalled")
                else:
                    print_warning("No existing installation to uninstall")
        except:
            pass

        # Clear pip cache for introdl specifically
        if verbose:
            print_info("Clearing pip cache for introdl...")
        try:
            # This removes any cached wheels for introdl
            subprocess.run([sys.executable, "-m", "pip", "cache", "remove", "introdl"],
                          capture_output=True, text=True)
        except:
            # pip cache command might not be available in older versions
            pass

        # Verify the source directory structure before installing (only in verbose)
        if verbose:
            print_info("Verifying source directory...")
            src_init = introdl_dir / "src" / "introdl" / "__init__.py"
            if src_init.exists():
                print_status(f"Source __init__.py found at: {src_init}")
                # Double-check version in source
                with open(src_init, 'r') as f:
                    for line in f:
                        if '__version__' in line:
                            print_info(f"Source file contains: {line.strip()}")
                            break
            else:
                print_error(f"Source __init__.py not found at expected location: {src_init}")

            # Check if setup.py or pyproject.toml exists
            setup_py = introdl_dir / "setup.py"
            pyproject = introdl_dir / "pyproject.toml"
            if setup_py.exists():
                print_status("Found setup.py")
            if pyproject.exists():
                print_status("Found pyproject.toml")

        # Install fresh - use no-cache-dir to prevent using cached version
        if verbose:
            print_info("Installing new version (bypassing cache)...")
            print_info(f"Installing from: {introdl_dir}")

        try:
            # Use no-cache-dir to force pip to use the actual source files
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", str(introdl_dir), "--no-cache-dir", "--upgrade"
            ], capture_output=True, text=True)

            # Show what pip is doing (only in verbose)
            if verbose and result.stdout:
                output_lines = result.stdout.split('\n')[:5]
                for line in output_lines:
                    if line.strip():
                        print_info(f"pip: {line.strip()}")

            if result.returncode == 0:
                if verbose:
                    print_status("Installation command completed")

                # Wait a moment for installation to complete
                import time
                time.sleep(1)

                # Verify the new version was actually installed
                if verbose:
                    print_info("Verifying installation...")

                # Check if module can be imported at all
                import_check = subprocess.run([
                    sys.executable, "-c", "import introdl; print('Import successful')"
                ], capture_output=True, text=True)

                if import_check.returncode != 0:
                    print_warning("Cannot import introdl after installation")
                    if verbose:
                        print_info(f"Import error: {import_check.stderr.strip()}")

                    # Try to find where it was installed (verbose only)
                    if verbose:
                        pip_show = subprocess.run([
                            sys.executable, "-m", "pip", "show", "introdl"
                        ], capture_output=True, text=True)

                        if pip_show.returncode == 0:
                            print_info("Package is installed according to pip:")
                            for line in pip_show.stdout.split('\n')[:5]:
                                if line.strip():
                                    print_info(f"  {line.strip()}")
                        else:
                            print_error("Package not found by pip show")

                else:
                    # Import successful, check version
                    version_check = subprocess.run([
                        sys.executable, "-c", "import introdl; print(introdl.__version__)"
                    ], capture_output=True, text=True)

                    if version_check.returncode == 0:
                        new_version = version_check.stdout.strip()
                        if new_version == source_version:
                            print_status(f"Successfully updated to introdl v{new_version}")
                        else:
                            print_warning(f"Installation completed but version is {new_version}, expected {source_version}")
                            print_info("Try restarting the kernel and running again", force=True)

                    # Test the installation (verbose only)
                    if verbose:
                        test_result = subprocess.run([
                            sys.executable, "-c", "from introdl.utils import config_paths_keys; print('Works!')"
                        ], capture_output=True, text=True)

                        if test_result.returncode == 0:
                            print_status("Installation verified - introdl.utils imports correctly")
                        else:
                            print_warning("Installation succeeded but imports may need kernel restart")
                            if test_result.stderr:
                                print_info(f"Import error: {test_result.stderr.strip()}")

                print("\n🔄 IMPORTANT: Restart your kernel and run this cell again!")
                print("=" * 57)
                sys.exit(2)  # Special exit code to indicate restart needed

            else:
                print_error(f"Installation failed with error code {result.returncode}")
                if result.stderr:
                    print_error(f"Error message: {result.stderr}")
                print_info("Trying alternative installation method...", force=True)

                # Try without upgrade flag and with no-cache-dir
                result2 = subprocess.run([
                    sys.executable, "-m", "pip", "install", str(introdl_dir), "--force-reinstall", "--no-cache-dir"
                ], capture_output=True, text=True)

                if result2.returncode == 0:
                    print_status("Alternative installation succeeded")
                    print("\n🔄 IMPORTANT: Restart your kernel and run this cell again!")
                    print("=" * 57)
                    sys.exit(2)
                else:
                    print_error("Alternative installation also failed")
                    if result2.stderr:
                        print_error(f"Error: {result2.stderr}")
                    print("📝 Fallback: Try running the full Course_Setup.ipynb notebook")
                    sys.exit(1)

        except Exception as e:
            print_error(f"Installation error: {e}")
            sys.exit(1)

    # Quick health checks (only if no update was needed and in verbose mode)
    if verbose:
        print("\n🔧 Quick health check...")

        # Check introdl.utils import
        try:
            subprocess.run([sys.executable, "-c", "from introdl.utils import config_paths_keys"],
                          capture_output=True, text=True, check=True)
            print_status("introdl.utils imports correctly")
        except:
            print_error("introdl.utils import failed - may need kernel restart")

    # Check workspace setup (always)
    home_workspace = course_root / "home_workspace"
    home_workspace.mkdir(parents=True, exist_ok=True)

    # Check API keys file (silent unless missing)
    api_keys_file = home_workspace / "api_keys.env"
    if not api_keys_file.exists():
        api_keys_template = script_dir / "api_keys.env"
        if api_keys_template.exists():
            import shutil
            shutil.copy2(api_keys_template, api_keys_file)
            print_info("Created API keys template at home_workspace/api_keys.env", force=True)
            print("📝 Remember to add your actual API keys for Lessons 7+")
        else:
            print_warning("API keys template not found")
    elif verbose:
        print_status("API keys file exists")

    # Check if we're in a local git repo (only in verbose)
    if verbose and (course_root / ".git").exists():
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
    elif verbose:
        print_info("Cloud environment (CoCalc/Colab) detected")

    # Final message only shown when already up to date and not verbose
    if not needs_update and not verbose:
        # Message already shown above: "introdl vX.X.X already up to date"
        pass

if __name__ == "__main__":
    main()