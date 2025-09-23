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

def main():
    global verbose
    # Check for verbose flag
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

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
        if not verbose:
            # Simple message for non-verbose mode
            print("📦 Updating introdl to v{}...".format(source_version))
        else:
            print("\n📦 Installing/updating introdl...")
            print("=" * 32)

        # CRITICAL: Find the ACTUAL installation location using pip show
        # This is the most reliable way to find where the package is really installed
        actual_install_location = None
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "show", "introdl"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Location:'):
                        actual_install_location = Path(line.split(':', 1)[1].strip())
                        if verbose:
                            print_info(f"Found introdl installed at: {actual_install_location}")
                        break
        except:
            pass

        # If we found the actual location, check if it has the old structure and remove it
        if actual_install_location and actual_install_location.exists():
            introdl_path = actual_install_location / "introdl"
            if introdl_path.exists():
                # Check for old nested structure
                old_subdirs = ['utils', 'nlp', 'idlmam', 'visul']
                has_old_structure = any((introdl_path / subdir).exists() for subdir in old_subdirs)

                if has_old_structure:
                    print_warning(f"⚠️  OLD nested structure detected at: {introdl_path}")
                    print_info("Aggressively removing old package and subdirectories...", force=True)

                    # First try to remove just the old subdirectories
                    old_subdirs = ['utils', 'nlp', 'idlmam', 'visul']
                    for subdir in old_subdirs:
                        subdir_path = introdl_path / subdir
                        if subdir_path.exists():
                            try:
                                import shutil
                                shutil.rmtree(subdir_path)
                                print_info(f"  Removed old subdir: {subdir}/", force=True)
                            except PermissionError:
                                # Try to at least delete the __init__.py files to break imports
                                try:
                                    init_file = subdir_path / "__init__.py"
                                    if init_file.exists():
                                        init_file.unlink()
                                        print_info(f"  Deleted {subdir}/__init__.py to break imports", force=True)

                                    # Also try to rename the directory to break imports
                                    try:
                                        subdir_path.rename(subdir_path.parent / f"_old_{subdir}")
                                        print_info(f"  Renamed {subdir}/ to _old_{subdir}/ to break imports", force=True)
                                    except:
                                        pass
                                except Exception as e2:
                                    print_warning(f"  Could not fully remove {subdir}: {e2}")
                            except Exception as e:
                                print_warning(f"  Error removing {subdir}: {e}")

                    # Now try to remove the entire introdl directory
                    try:
                        import shutil
                        shutil.rmtree(introdl_path)
                        print_status(f"✅ Completely removed old package from: {actual_install_location}")
                    except PermissionError as e:
                        print_error(f"Permission denied removing {introdl_path}")
                        print_error("⚠️  MANUAL INTERVENTION REQUIRED!")
                        print_info("Please run this command manually:", force=True)
                        print_info(f"  sudo rm -rf {introdl_path}", force=True)
                        print_info("Then restart the kernel and run this cell again.", force=True)
                        sys.exit(1)  # Exit to force user to handle this
                    except Exception as e:
                        print_warning(f"Could not fully remove {introdl_path}: {e}")
                        # Continue anyway - maybe partial removal will help

                # Also check for and remove egg-info
                for egg_info in actual_install_location.glob("introdl*.egg-info"):
                    try:
                        import shutil
                        shutil.rmtree(egg_info)
                        if verbose:
                            print_info(f"Removed egg-info: {egg_info}")
                    except:
                        pass

        # Also check common locations like /usr/local/lib/python*/dist-packages
        # This is especially important for Hyperstack
        common_locations = [
            Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
            Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
            Path(site.getusersitepackages()),
        ]

        for location in common_locations:
            if location.exists():
                introdl_path = location / "introdl"
                if introdl_path.exists():
                    # Check for old structure
                    old_subdirs = ['utils', 'nlp', 'idlmam', 'visul']
                    has_old_structure = any((introdl_path / subdir).exists() for subdir in old_subdirs)

                    if has_old_structure:
                        print_warning(f"OLD structure found at: {location}")

                        # Aggressively remove subdirectories
                        for subdir in old_subdirs:
                            subdir_path = introdl_path / subdir
                            if subdir_path.exists():
                                try:
                                    import shutil
                                    shutil.rmtree(subdir_path)
                                    print_info(f"  Removed: {subdir}/", force=True)
                                except PermissionError:
                                    # At minimum, try to break imports
                                    try:
                                        init_file = subdir_path / "__init__.py"
                                        if init_file.exists():
                                            init_file.unlink()
                                            print_info(f"  Deleted {subdir}/__init__.py", force=True)
                                    except:
                                        pass
                                    try:
                                        subdir_path.rename(subdir_path.parent / f"_broken_{subdir}")
                                        print_info(f"  Renamed to _broken_{subdir}/", force=True)
                                    except:
                                        pass
                                except Exception as e:
                                    if verbose:
                                        print_warning(f"  Could not remove {subdir}: {e}")

                        # Try to remove entire directory
                        try:
                            import shutil
                            shutil.rmtree(introdl_path)
                            print_status(f"Removed entire old package from: {location}")
                        except PermissionError:
                            print_error(f"⚠️  PERMISSION DENIED at {introdl_path}")
                            print_error("Manual removal required - run:")
                            print_info(f"  sudo rm -rf {introdl_path}", force=True)
                        except Exception as e:
                            if verbose:
                                print_warning(f"Could not fully remove {introdl_path}: {e}")

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