#!/usr/bin/env python3
"""
Aggressive cleanup script for introdl on Hyperstack compute servers.

This script performs a complete removal of ALL introdl installations and artifacts,
then does a fresh install. Designed specifically for Hyperstack environment where
old package structures persist despite normal cleanup attempts.

Usage:
    python3 hyperstack_clean_introdl.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import site


def print_status(msg):
    print(f"✅ {msg}")


def print_warning(msg):
    print(f"⚠️  {msg}")


def print_error(msg):
    print(f"❌ {msg}")


def print_info(msg):
    print(f"ℹ️  {msg}")


def remove_directory(path, description=""):
    """Aggressively remove a directory with multiple fallback strategies."""
    if not path.exists():
        return True

    try:
        # First attempt: normal removal
        shutil.rmtree(path)
        print_status(f"Removed {description}: {path}")
        return True
    except PermissionError:
        print_warning(f"Permission denied for {path}, trying alternative methods...")

        # Second attempt: Remove all __init__.py files to break imports
        try:
            for init_file in path.rglob("__init__.py"):
                try:
                    init_file.unlink()
                    print_info(f"  Deleted {init_file.relative_to(path.parent)}")
                except:
                    pass
        except:
            pass

        # Third attempt: Rename directory to break imports
        try:
            broken_path = path.parent / f"_BROKEN_{path.name}"
            path.rename(broken_path)
            print_warning(f"  Renamed to {broken_path.name} to break imports")
            return True
        except:
            pass

        # Fourth attempt: Empty all Python files
        try:
            for py_file in path.rglob("*.py"):
                try:
                    py_file.write_text("# File disabled by cleanup script\n")
                    print_info(f"  Emptied {py_file.relative_to(path.parent)}")
                except:
                    pass
        except:
            pass

        print_error(f"  Could not fully remove {path}")
        print_error(f"  Manual intervention required: sudo rm -rf {path}")
        return False
    except Exception as e:
        print_error(f"Error removing {path}: {e}")
        return False


def main():
    print("=" * 60)
    print("🧹 AGGRESSIVE introdl Cleanup for Hyperstack")
    print("=" * 60)

    # Get paths
    script_dir = Path(__file__).parent.absolute()
    course_root = script_dir.parent.parent
    introdl_source = script_dir / "introdl"

    if not introdl_source.exists():
        print_error(f"Source introdl not found at {introdl_source}")
        sys.exit(1)

    print(f"Source location: {introdl_source}")
    print("")

    # Step 1: Clean ALL build artifacts from source directory
    print("1️⃣  Cleaning source directory build artifacts...")
    artifacts_to_clean = [
        introdl_source / "build",
        introdl_source / "dist",
        introdl_source / "src" / "introdl.egg-info",
        introdl_source / "introdl.egg-info",
        introdl_source / "__pycache__",
        introdl_source / "src" / "introdl" / "__pycache__",
    ]

    for artifact in artifacts_to_clean:
        if artifact.exists():
            remove_directory(artifact, "build artifact")

    # Also remove all __pycache__ directories
    for pycache in introdl_source.rglob("__pycache__"):
        remove_directory(pycache, "cache")

    # Remove all .pyc files
    for pyc in introdl_source.rglob("*.pyc"):
        try:
            pyc.unlink()
            print_info(f"  Removed {pyc.name}")
        except:
            pass

    print("")

    # Step 2: Find ALL possible installation locations
    print("2️⃣  Finding all introdl installations...")
    locations_to_check = []

    # Standard site-packages
    try:
        locations_to_check.extend(site.getsitepackages())
    except:
        pass

    # User site-packages
    try:
        locations_to_check.append(site.getusersitepackages())
    except:
        pass

    # Virtual environment locations
    locations_to_check.extend([
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "dist-packages",
        Path(sys.prefix) / "local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(sys.prefix) / "local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "dist-packages",
    ])

    # Hyperstack-specific locations
    locations_to_check.extend([
        Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
        Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"),
        Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
        Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"),
        Path("/opt/conda/lib/python3.10/site-packages"),  # Common in containers
        Path("/opt/conda/lib/python3.11/site-packages"),
        Path("/opt/conda/lib/python3.9/site-packages"),
    ])

    # Add paths from sys.path
    for p in sys.path:
        if 'packages' in p:
            locations_to_check.append(Path(p))

    # Remove duplicates and non-existent paths
    locations_to_check = list(set(Path(loc) for loc in locations_to_check if Path(loc).exists()))

    print(f"Checking {len(locations_to_check)} locations...")
    print("")

    # Step 3: Remove ALL introdl installations
    print("3️⃣  Removing ALL introdl installations...")
    found_installations = []
    old_structure_found = False

    for location in locations_to_check:
        introdl_path = location / "introdl"
        if introdl_path.exists():
            found_installations.append(introdl_path)

            # Check for old structure
            old_subdirs = ['utils', 'nlp', 'idlmam', 'visul']
            has_old = any((introdl_path / subdir).exists() for subdir in old_subdirs)

            if has_old:
                old_structure_found = True
                print_warning(f"OLD STRUCTURE found at: {introdl_path}")

                # Remove old subdirectories first
                for subdir in old_subdirs:
                    subdir_path = introdl_path / subdir
                    if subdir_path.exists():
                        remove_directory(subdir_path, f"old {subdir}")

            # Remove the entire introdl directory
            remove_directory(introdl_path, "introdl package")

        # Also check for egg-info
        for egg in location.glob("introdl*.egg-info"):
            remove_directory(egg, "egg-info")

        # Check for dist-info
        for dist in location.glob("introdl*.dist-info"):
            remove_directory(dist, "dist-info")

    if not found_installations:
        print_info("No existing installations found")
    else:
        print_info(f"Found and cleaned {len(found_installations)} installation(s)")

    print("")

    # Step 4: Clear Python's module cache
    print("4️⃣  Clearing Python module cache...")
    if 'introdl' in sys.modules:
        del sys.modules['introdl']
        print_info("  Removed introdl from sys.modules")

    # Remove all introdl submodules from cache
    modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('introdl.')]
    for mod in modules_to_remove:
        del sys.modules[mod]
        print_info(f"  Removed {mod} from sys.modules")

    print("")

    # Step 5: Multiple pip uninstalls to catch any remaining metadata
    print("5️⃣  Running pip uninstall (multiple times)...")
    for i in range(3):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "introdl", "-y"],
                capture_output=True, text=True, timeout=10
            )
            if "not installed" in result.stdout.lower():
                print_info(f"  Round {i+1}: Nothing to uninstall")
                break
            else:
                print_info(f"  Round {i+1}: Uninstalled")
        except:
            pass

    print("")

    # Step 6: Clear pip cache
    print("6️⃣  Clearing pip cache...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "cache", "purge"],
            capture_output=True, text=True, timeout=10
        )
        print_status("Pip cache cleared")
    except:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "cache", "remove", "introdl"],
                capture_output=True, text=True, timeout=10
            )
            print_status("Removed introdl from pip cache")
        except:
            print_warning("Could not clear pip cache (may not be available)")

    print("")

    # Step 7: Fresh install with no cache
    print("7️⃣  Installing fresh introdl package...")
    print(f"   Installing from: {introdl_source}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", str(introdl_source),
             "--no-cache-dir", "--force-reinstall", "--no-deps"],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            print_status("Installation command completed")

            # Install dependencies separately
            print_info("Installing dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", str(introdl_source),
                 "--no-cache-dir"],
                capture_output=True, text=True, timeout=30
            )

        else:
            print_error("Installation failed")
            print(result.stderr)
            sys.exit(1)

    except Exception as e:
        print_error(f"Installation error: {e}")
        sys.exit(1)

    print("")

    # Step 8: Verify installation
    print("8️⃣  Verifying installation...")

    # Check version
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import introdl; print(f'Version: {introdl.__version__}')"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print_status(result.stdout.strip())
        else:
            print_warning("Could not verify version")
    except:
        print_warning("Version check timed out")

    # Check for old structure in new installation
    print_info("Checking for old structure in new installation...")
    new_install_clean = True

    for location in locations_to_check:
        introdl_path = location / "introdl"
        if introdl_path.exists():
            old_subdirs = ['utils', 'nlp', 'idlmam', 'visul']
            for subdir in old_subdirs:
                if (introdl_path / subdir).exists():
                    print_error(f"⚠️  OLD STRUCTURE STILL EXISTS: {introdl_path / subdir}")
                    new_install_clean = False

    if new_install_clean:
        print_status("No old structure found in new installation")
    else:
        print_error("OLD STRUCTURE PERSISTS - Manual intervention required!")
        print_error("Try running with sudo: sudo python3 hyperstack_clean_introdl.py")

    print("")
    print("=" * 60)
    if new_install_clean:
        print("✅ CLEANUP COMPLETE - Restart kernel and try again")
    else:
        print("⚠️  PARTIAL SUCCESS - Old structure may persist")
        print("   Try: sudo python3 hyperstack_clean_introdl.py")
    print("=" * 60)

    # Exit with special code to indicate kernel restart needed
    sys.exit(2 if new_install_clean else 1)


if __name__ == "__main__":
    main()