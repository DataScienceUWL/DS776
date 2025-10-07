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
import re
import site

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

def get_installed_version():
    """Get the installed version without importing the module (faster and avoids timeout)."""
    try:
        # First try to directly read from common installation locations
        possible_locations = [
            Path(site.getusersitepackages()),  # User site-packages
            Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
            Path(sys.prefix) / "local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        ]

        # Add any paths from sys.path that contain site-packages
        for p in sys.path:
            if 'site-packages' in p and Path(p).exists():
                possible_locations.append(Path(p))

        # Check each location for introdl
        for location in possible_locations:
            init_file = location / "introdl" / "__init__.py"
            if init_file.exists():
                try:
                    with open(init_file, 'r') as f:
                        for line in f:
                            if '__version__' in line and '=' in line:
                                match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', line)
                                if match:
                                    version = match.group(1)
                                    if verbose:
                                        print_info(f"Found introdl in {location}")
                                    return version, str(location)
                except:
                    continue

        # Fallback to pip show if direct search fails (with longer timeout)
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "show", "introdl"
            ], capture_output=True, text=True, timeout=15)  # Increased timeout

            if result.returncode == 0:
                # Parse the output to get location and version
                location = None
                version = None
                for line in result.stdout.split('\n'):
                    if line.startswith('Location:'):
                        location = line.split(':', 1)[1].strip()
                    elif line.startswith('Version:'):
                        version = line.split(':', 1)[1].strip()

                if location and version:
                    # Verify the version by reading __init__.py directly
                    init_file = Path(location) / "introdl" / "__init__.py"
                    if init_file.exists():
                        try:
                            with open(init_file, 'r') as f:
                                for line in f:
                                    if '__version__' in line and '=' in line:
                                        match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', line)
                                        if match:
                                            file_version = match.group(1)
                                            # Use the version from the file if found
                                            if file_version:
                                                version = file_version
                                            break
                        except:
                            pass  # Use pip version if we can't read the file

                    return version, location
        except subprocess.TimeoutExpired:
            if verbose:
                print_warning("pip show timed out, package may still be installed")
        except Exception as e:
            if verbose:
                print_info(f"pip show failed: {e}")

        return None, None

    except Exception as e:
        if verbose:
            print_error(f"Error checking installation: {e}")
        return None, None

def is_hyperstack():
    """Detect if we're running on Hyperstack compute server."""
    hostname = os.environ.get("HOSTNAME", "").lower()

    # CoCalc has hostnames like "project-..." or contains "cocalc"
    if "cocalc" in hostname or hostname.startswith("project-"):
        return False  # Definitely CoCalc, not Hyperstack

    # Hyperstack-specific: hostname like "prod-9684" AND CUDA environment
    is_hyperstack_hostname = "prod-" in hostname and any(char.isdigit() for char in hostname)
    has_nvidia_env = "NVIDIA_VISIBLE_DEVICES" in os.environ or "CUDA_VERSION" in os.environ

    # Hyperstack has very specific combination:
    # 1. prod-XXXX hostname pattern
    # 2. NVIDIA/CUDA environment variables
    # 3. /workspace directory exists
    # 4. Running in a Docker container
    if is_hyperstack_hostname and has_nvidia_env:
        # Double-check with workspace and docker
        if os.path.exists("/workspace") and os.path.exists("/.dockerenv"):
            return True

    return False


def clean_source_build_artifacts(introdl_dir, verbose_mode=False):
    """Clean build artifacts from the source directory."""
    if verbose_mode:
        print_info("Cleaning source directory build artifacts...", force=True)

    artifacts = [
        introdl_dir / "build",
        introdl_dir / "dist",
        introdl_dir / "src" / "introdl.egg-info",
        introdl_dir / "introdl.egg-info",
    ]

    for artifact in artifacts:
        if artifact.exists():
            try:
                import shutil
                shutil.rmtree(artifact)
                if verbose_mode:
                    print_info(f"  Removed {artifact.name}", force=True)
            except Exception as e:
                if verbose:
                    print_warning(f"  Could not remove {artifact.name}: {e}")

    # Remove all __pycache__ directories
    for pycache in introdl_dir.rglob("__pycache__"):
        try:
            import shutil
            shutil.rmtree(pycache)
            if verbose:
                print_info(f"  Removed cache: {pycache.name}")
        except:
            pass

    # Remove all .pyc files (bytecode)
    try:
        import subprocess
        subprocess.run(["find", str(introdl_dir), "-name", "*.pyc", "-delete"],
                      capture_output=True, timeout=5)
    except:
        pass


def clean_all_installations():
    """Comprehensively clean introdl from all possible installation locations."""
    import shutil

    # Build list of all possible site-packages locations
    locations = [
        Path(site.getusersitepackages()),
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(sys.prefix) / "local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
        Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
        Path("/usr/lib/python3/dist-packages"),
    ]

    # Add paths from sys.path
    for p in sys.path:
        if 'site-packages' in p or 'dist-packages' in p:
            path_obj = Path(p)
            if path_obj.exists():
                locations.append(path_obj)

    # Remove duplicates
    locations = list(set(loc for loc in locations if loc.exists()))

    cleaned_any = False
    for location in locations:
        introdl_path = location / "introdl"

        # Remove introdl directory
        if introdl_path.exists():
            try:
                shutil.rmtree(introdl_path)
                if verbose:
                    print_info(f"Removed introdl from: {location}")
                cleaned_any = True
            except PermissionError:
                print_warning(f"Permission denied: {introdl_path}")
                print_error("Manual cleanup required - run:")
                print_info(f"  sudo rm -rf {introdl_path}", force=True)
            except Exception as e:
                if verbose:
                    print_warning(f"Could not remove {introdl_path}: {e}")

        # Remove egg-info and dist-info directories
        for info_dir in location.glob("introdl*.egg-info"):
            try:
                shutil.rmtree(info_dir)
                if verbose:
                    print_info(f"Removed: {info_dir.name}")
                cleaned_any = True
            except:
                pass

        for info_dir in location.glob("introdl*.dist-info"):
            try:
                shutil.rmtree(info_dir)
                if verbose:
                    print_info(f"Removed: {info_dir.name}")
                cleaned_any = True
            except:
                pass

    if cleaned_any and verbose:
        print_status("Cleaned all installation locations")

    return cleaned_any


def clean_course_tools_directory(course_tools_dir, force=False):
    """
    Clean obsolete files from Course_Tools directory.
    Only runs on first execution or when forced (via --verbose).

    Removes:
    - Files with ~ suffix (CoCalc backup files)
    - Obsolete scripts from old versions
    - __pycache__ directory in Course_Tools root

    Uses timestamp file to avoid repeated cleanup (runs max once per 7 days).
    """
    import time
    import shutil

    # Check timestamp file to avoid repeated cleanup
    timestamp_file = course_tools_dir / ".last_cleanup"

    # Skip if cleaned recently (within 7 days) unless forced
    if not force and timestamp_file.exists():
        try:
            last_cleanup = timestamp_file.stat().st_mtime
            days_since = (time.time() - last_cleanup) / 86400
            if days_since < 7:
                return False  # Skip cleanup
        except:
            pass  # If we can't read timestamp, proceed with cleanup

    cleaned_any = False

    # Remove all files with ~ suffix (CoCalc backup files)
    for tilde_file in course_tools_dir.glob("*~"):
        try:
            tilde_file.unlink()
            if verbose:
                print_info(f"Removed backup: {tilde_file.name}")
            cleaned_any = True
        except Exception as e:
            if verbose:
                print_warning(f"Could not remove {tilde_file.name}: {e}")

    # Remove specific obsolete files (from old package structure migration)
    obsolete_files = [
        "diagnose_hyperstack.py",
        "diagnose_introdl.py",
        "force_clean_introdl.py",
        "hyperstack_clean_introdl.py",
        "update_introdl.py",
        "update_introdl.sh",
        "setup_course.sh",
        "To_Do_List.ipynb"
    ]

    for filename in obsolete_files:
        filepath = course_tools_dir / filename
        if filepath.exists():
            try:
                filepath.unlink()
                if verbose:
                    print_info(f"Removed obsolete: {filename}")
                cleaned_any = True
            except Exception as e:
                if verbose:
                    print_warning(f"Could not remove {filename}: {e}")

    # Remove __pycache__ in Course_Tools root (not in introdl/)
    pycache = course_tools_dir / "__pycache__"
    if pycache.exists():
        try:
            shutil.rmtree(pycache)
            if verbose:
                print_info("Removed __pycache__ from Course_Tools")
            cleaned_any = True
        except Exception as e:
            if verbose:
                print_warning(f"Could not remove __pycache__: {e}")

    # Update timestamp file
    if cleaned_any or force:
        try:
            timestamp_file.touch()
        except:
            pass  # Don't fail if we can't write timestamp

    return cleaned_any


def main():
    global verbose
    # Check for verbose flag
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    # Detect if we're on Hyperstack
    on_hyperstack = is_hyperstack()
    if on_hyperstack:
        print_warning("🖥️  Hyperstack environment detected - using aggressive cleanup")
        verbose = True  # Force verbose mode on Hyperstack for better debugging

    # Quick exit if user wants to skip the check
    if os.environ.get('SKIP_INTRODL_CHECK', '').lower() in ('1', 'true', 'yes'):
        if verbose:
            print("Skipping introdl check (SKIP_INTRODL_CHECK is set)")
        return

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

    # CRITICAL: Clean source build artifacts FIRST, before any version detection
    # This prevents pip from using stale build artifacts that may have old structure
    clean_source_build_artifacts(introdl_dir, verbose_mode=False)

    # Clean Course_Tools directory of obsolete files (runs max once per 7 days)
    cleaned_course_tools = clean_course_tools_directory(script_dir, force=verbose)
    if cleaned_course_tools and not verbose:
        print_status("Cleaned Course_Tools directory")

    # Get source version from __init__.py
    source_version = None
    init_file = introdl_dir / "src" / "introdl" / "__init__.py"
    if init_file.exists():
        try:
            with open(init_file, 'r') as f:
                for line in f:
                    if '__version__' in line and '=' in line:
                        # Extract version string more carefully
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

    # Check installed version (fast method without importing)
    installed_version, install_location = get_installed_version()

    if verbose and installed_version:
        print_status(f"Installed version: {installed_version}")
        print_info(f"Location: {install_location}")
    elif verbose:
        print_warning("introdl not installed or not detectable")

    # Check for old nested module structure FIRST
    # This is critical for Hyperstack where old structure causes import errors
    has_old_structure = False
    old_structure_location = None

    # Check using pip show first
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", "introdl"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Location:'):
                    pip_location = Path(line.split(':', 1)[1].strip())
                    introdl_path = pip_location / "introdl"
                    if introdl_path.exists():
                        # Check for old subdirectories
                        old_subdirs = ['utils', 'nlp', 'idlmam', 'visul']
                        for subdir in old_subdirs:
                            if (introdl_path / subdir).exists():
                                has_old_structure = True
                                old_structure_location = introdl_path
                                if verbose:
                                    print_warning(f"Found OLD nested structure at: {introdl_path}")
                                break
                    break
    except:
        pass

    # Also check common Hyperstack location
    if not has_old_structure:
        hyperstack_locations = [
            Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages/introdl"),
            Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages/introdl"),
        ]
        for introdl_path in hyperstack_locations:
            if introdl_path.exists():
                old_subdirs = ['utils', 'nlp', 'idlmam', 'visul']
                for subdir in old_subdirs:
                    if (introdl_path / subdir).exists():
                        has_old_structure = True
                        old_structure_location = introdl_path
                        if verbose:
                            print_warning(f"Found OLD nested structure at: {introdl_path}")
                        break
                if has_old_structure:
                    break

    # Version comparison logic
    needs_update = False

    if has_old_structure:
        # FORCE UPDATE if old structure exists
        print_warning("⚠️  OLD package structure detected - forcing complete reinstall")
        needs_update = True
    elif not installed_version:
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

    # Build list of search paths for installations
    # This will be used if we need to do a complete removal
    search_paths = [
        Path(site.getusersitepackages()),  # User site-packages
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(sys.prefix) / "local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]

    # Add any paths from sys.path that contain site-packages
    for p in sys.path:
        if 'site-packages' in p and Path(p).exists():
            search_paths.append(Path(p))

    # Remove duplicates and keep only existing paths
    search_paths = list(set(p for p in search_paths if p and p.exists()))

    # Install/update if needed
    if needs_update:
        # Clean source build artifacts again in verbose mode (already done silently above)
        if verbose or on_hyperstack:
            clean_source_build_artifacts(introdl_dir, verbose_mode=True)

        if not verbose:
            # Simple message for non-verbose mode
            print("📦 Updating introdl to v{}...".format(source_version))
        else:
            print("\n📦 Installing/updating introdl...")
            print("=" * 32)

        # Comprehensive cleanup of all installations
        if verbose:
            print_info("Removing old installations from all locations...")

        # Use our comprehensive cleanup function
        cleaned = clean_all_installations()

        if cleaned or has_old_structure:
            if not verbose:
                print_status("Removed old package installations")

        # Clear Python's module cache on Hyperstack
        if on_hyperstack:
            print_info("Clearing Python module cache (Hyperstack mode)...", force=True)
            if 'introdl' in sys.modules:
                del sys.modules['introdl']
                print_info("  Removed introdl from sys.modules", force=True)

            # Remove all introdl submodules from cache
            modules_to_remove = [mod for mod in list(sys.modules.keys()) if mod.startswith('introdl.')]
            for mod in modules_to_remove:
                del sys.modules[mod]
                if verbose:
                    print_info(f"  Removed {mod} from sys.modules", force=True)

        # Now use pip uninstall to clean up any metadata
        if verbose:
            print_info("Running pip uninstall to clean metadata...")
        try:
            # Run pip uninstall multiple times in case there are multiple installations
            for _ in range(3):  # Try up to 3 times
                uninstall_result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "introdl", "-y"],
                             capture_output=True, text=True, timeout=10)
                if "Cannot uninstall" not in uninstall_result.stdout and "not installed" in uninstall_result.stdout.lower():
                    break  # Nothing left to uninstall
        except:
            pass

        # Clear pip cache - more aggressive on Hyperstack
        if on_hyperstack:
            print_info("Purging entire pip cache (Hyperstack mode)...", force=True)
            try:
                # Try to purge entire cache on Hyperstack
                subprocess.run([sys.executable, "-m", "pip", "cache", "purge"],
                              capture_output=True, text=True, timeout=10)
                print_status("Pip cache purged")
            except:
                # Fallback to removing just introdl
                try:
                    subprocess.run([sys.executable, "-m", "pip", "cache", "remove", "introdl"],
                                  capture_output=True, text=True)
                except:
                    pass
        else:
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

                # Check the installed version using our fast method
                try:
                    new_version, new_location = get_installed_version()
                except Exception as e:
                    if verbose:
                        print_info(f"Error verifying installation: {e}")
                    new_version = None

                if new_version:
                    if new_version == source_version:
                        print_status(f"Successfully updated to introdl v{new_version}")
                    else:
                        print_warning(f"Installation completed but version is {new_version}, expected {source_version}")
                        print_info("Try restarting the kernel and running again", force=True)

                    # Skip import test - it often times out due to heavy module loading
                    # The version check above is sufficient to verify installation
                else:
                    print_warning("Cannot verify installation - may need kernel restart")

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

        except subprocess.TimeoutExpired as e:
            print_warning("Installation process timed out")
            print_info("The installation likely succeeded but verification timed out", force=True)
            print("\n🔄 IMPORTANT: Restart your kernel and run this cell again!")
            print("=" * 57)
            sys.exit(2)
        except Exception as e:
            print_error(f"Installation error: {e}")
            sys.exit(1)

    # Skip health checks - imports often timeout due to heavy module loading

    # Only do workspace setup if we did an update or in verbose mode
    if needs_update or verbose:
        # Check workspace setup
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