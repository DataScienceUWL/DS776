#!/usr/bin/env python3
"""
Update introdl package - removes old package structure and installs the newest version.

This script:
1. Removes old subdirectory structure (idlmam/, utils/, visul/, nlp/) from introdl/src/introdl/
2. Removes all build artifacts (build/, dist/, *.egg-info, __pycache__)
3. Uninstalls any existing introdl package
4. Installs the fresh version from Lessons/Course_Tools/introdl

Usage:
    python update_introdl.py
    or
    python3 update_introdl.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def run_command(cmd, ignore_errors=False):
    """Run a shell command and handle errors."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and not ignore_errors:
        print(f"Error: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True


def main():
    print("=" * 60)
    print("introdl Package Update Script")
    print("=" * 60)

    # Get the script directory (Course_Tools)
    script_dir = Path(__file__).parent.resolve()
    introdl_dir = script_dir / "introdl"

    if not introdl_dir.exists():
        print(f"Error: introdl package not found at {introdl_dir}")
        sys.exit(1)

    print(f"\nWorking with introdl package at: {introdl_dir}")

    # Step 1: Clean old subdirectory structure
    print("\n1. Removing old package subdirectories...")
    old_subdirs = ["idlmam", "utils", "visul", "nlp"]
    src_introdl = introdl_dir / "src" / "introdl"

    if src_introdl.exists():
        for subdir in old_subdirs:
            old_path = src_introdl / subdir
            if old_path.exists() and old_path.is_dir():
                print(f"   Removing old directory: {old_path}")
                shutil.rmtree(old_path)

    # Also check in the root introdl directory (in case of different structure)
    for subdir in old_subdirs:
        old_path = introdl_dir / subdir
        if old_path.exists() and old_path.is_dir():
            print(f"   Removing old directory: {old_path}")
            shutil.rmtree(old_path)

    # Step 2: Clean build artifacts
    print("\n2. Cleaning build artifacts...")
    artifacts_to_remove = [
        "build",
        "dist",
        "*.egg-info",
        "src/*.egg-info",
        "src/introdl.egg-info"
    ]

    for artifact in artifacts_to_remove:
        # Handle wildcards
        if "*" in artifact:
            base_dir = introdl_dir
            if "/" in artifact:
                parts = artifact.split("/")
                base_dir = introdl_dir / "/".join(parts[:-1])
                pattern = parts[-1]
            else:
                pattern = artifact

            if base_dir.exists():
                for item in base_dir.glob(pattern):
                    if item.exists():
                        print(f"   Removing: {item}")
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
        else:
            artifact_path = introdl_dir / artifact
            if artifact_path.exists():
                print(f"   Removing: {artifact_path}")
                if artifact_path.is_dir():
                    shutil.rmtree(artifact_path)
                else:
                    artifact_path.unlink()

    # Step 3: Clean __pycache__ directories
    print("\n3. Cleaning __pycache__ directories...")
    for pycache in introdl_dir.rglob("__pycache__"):
        print(f"   Removing: {pycache}")
        shutil.rmtree(pycache)

    # Step 4: Remove .pyc and .pyo files
    print("\n4. Cleaning compiled Python files...")
    for ext in ["*.pyc", "*.pyo"]:
        for compiled_file in introdl_dir.rglob(ext):
            print(f"   Removing: {compiled_file}")
            compiled_file.unlink()

    # Step 5: Uninstall existing introdl package
    print("\n5. Uninstalling existing introdl package (if any)...")
    run_command("pip uninstall -y introdl", ignore_errors=True)

    # Step 6: Install fresh version
    print("\n6. Installing fresh introdl package...")
    os.chdir(introdl_dir)

    # Try pip install with editable mode first (for development)
    if not run_command("pip install -e ."):
        # If editable fails, try regular install
        print("   Editable install failed, trying regular install...")
        if not run_command("pip install ."):
            print("\nError: Failed to install introdl package")
            sys.exit(1)

    # Step 7: Verify installation
    print("\n7. Verifying installation...")
    verification_code = """
import sys
try:
    import introdl
    print(f"✓ introdl successfully imported")
    print(f"  Version: {introdl.__version__}")

    # Test imports of main modules
    from introdl import utils, idlmam, visul, nlp
    print("✓ All main modules imported successfully")

    # Show installation location
    import os
    print(f"  Installed at: {os.path.dirname(introdl.__file__)}")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
"""

    result = subprocess.run([sys.executable, "-c", verification_code],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        print("\n" + "=" * 60)
        print("SUCCESS: introdl package updated successfully!")
        print("=" * 60)
    else:
        print(result.stderr)
        print("\n" + "=" * 60)
        print("ERROR: Package installed but verification failed")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()