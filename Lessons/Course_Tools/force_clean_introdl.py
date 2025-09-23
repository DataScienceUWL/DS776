#!/usr/bin/env python3
"""
Force clean removal of old introdl package structure.
Run this if auto_update can't remove old directories due to permissions.
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil

def print_colored(symbol, message, color_code=None):
    """Print with color if supported"""
    if color_code and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        print(f"\033[{color_code}m{symbol}\033[0m {message}")
    else:
        print(f"{symbol} {message}")

def print_status(message):
    print_colored("✅", message, "0;32")

def print_warning(message):
    print_colored("⚠️", message, "1;33")

def print_error(message):
    print_colored("❌", message, "0;31")

def print_info(message):
    print_colored("ℹ️", message, "0;34")

print("=" * 60)
print("FORCE CLEAN OLD INTRODL PACKAGE")
print("=" * 60)

# Find all possible locations
locations_to_check = [
    Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
    Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
]

# Add location from pip show
try:
    result = subprocess.run([sys.executable, "-m", "pip", "show", "introdl"],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if line.startswith('Location:'):
                pip_location = Path(line.split(':', 1)[1].strip())
                locations_to_check.append(pip_location)
                break
except:
    pass

# Check each location
found_old = False
for location in locations_to_check:
    if not location.exists():
        continue

    introdl_path = location / "introdl"
    if not introdl_path.exists():
        continue

    print(f"\nChecking: {introdl_path}")

    # Check for old subdirectories
    old_subdirs = ['utils', 'nlp', 'idlmam', 'visul']
    old_found = []

    for subdir in old_subdirs:
        subdir_path = introdl_path / subdir
        if subdir_path.exists():
            old_found.append(subdir)
            found_old = True

    if old_found:
        print_warning(f"Found OLD subdirectories: {', '.join(old_found)}")

        # Try to remove with sudo if available
        if os.geteuid() == 0:  # Running as root
            print_info("Running as root, removing directly...")
            try:
                shutil.rmtree(introdl_path)
                print_status(f"Successfully removed: {introdl_path}")
            except Exception as e:
                print_error(f"Failed to remove: {e}")
        else:
            print_info("Not running as root. Attempting sudo removal...")

            # Try sudo rm -rf
            try:
                result = subprocess.run(
                    ["sudo", "rm", "-rf", str(introdl_path)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    print_status(f"Successfully removed with sudo: {introdl_path}")
                else:
                    print_error(f"Sudo removal failed: {result.stderr}")
            except FileNotFoundError:
                print_error("sudo not available")
                print_info("Please run manually as root:")
                print(f"  sudo rm -rf {introdl_path}")
            except subprocess.TimeoutExpired:
                print_error("Sudo command timed out")
            except Exception as e:
                print_error(f"Error running sudo: {e}")
                print_info("Please run manually:")
                print(f"  sudo rm -rf {introdl_path}")

        # Also clean egg-info
        for egg_info in location.glob("introdl*.egg-info"):
            print_info(f"Removing egg-info: {egg_info.name}")
            try:
                if os.geteuid() == 0:
                    shutil.rmtree(egg_info)
                else:
                    subprocess.run(["sudo", "rm", "-rf", str(egg_info)], timeout=5)
                print_status(f"Removed: {egg_info.name}")
            except:
                print_warning(f"Could not remove {egg_info.name}")

if found_old:
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print("Now run the auto_update script again to install fresh.")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("No old package structure found.")
    print("=" * 60)

# Final pip uninstall to clean metadata
print("\nRunning pip uninstall to clean any remaining metadata...")
for _ in range(3):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "introdl", "-y"],
            capture_output=True, text=True, timeout=10
        )
        if "not installed" in result.stdout.lower():
            break
    except:
        break

print("\n✅ Done! You can now run the auto_update script.")