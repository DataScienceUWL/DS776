#!/usr/bin/env python3
"""
Fix line endings for shell scripts in the Course_Tools directory.

This script converts Windows line endings (CRLF) to Unix line endings (LF)
for all .sh files in the current directory. This is necessary when cloning
the repository from GitHub on Windows and then using the scripts in a
Linux environment like CoCalc.

Usage:
    python fix_line_endings.py

After running this script, the shell scripts should work properly in Linux.
"""

import os
import glob
from pathlib import Path


def fix_line_endings(filepath):
    """
    Convert Windows line endings (CRLF) to Unix line endings (LF).

    Args:
        filepath: Path to the file to fix

    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        # Read the file in binary mode to preserve exact content
        with open(filepath, 'rb') as f:
            content = f.read()

        # Check if file has Windows line endings
        if b'\r\n' in content:
            # Replace CRLF with LF
            fixed_content = content.replace(b'\r\n', b'\n')

            # Write the fixed content back
            with open(filepath, 'wb') as f:
                f.write(fixed_content)

            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """
    Find and fix all shell scripts in the current directory.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    print("Fixing line endings for shell scripts in Course_Tools directory...")
    print(f"Working directory: {script_dir}")
    print()

    # Find all .sh files in the current directory
    shell_scripts = list(script_dir.glob("*.sh"))

    if not shell_scripts:
        print("No shell scripts (*.sh) found in the current directory.")
        return

    # Process each shell script
    modified_count = 0
    for script_path in shell_scripts:
        print(f"Checking: {script_path.name}")

        if fix_line_endings(script_path):
            print(f"  ✓ Fixed line endings for {script_path.name}")
            modified_count += 1
        else:
            print(f"  - {script_path.name} already has Unix line endings")

    print()
    if modified_count > 0:
        print(f"Successfully fixed line endings for {modified_count} file(s).")
        print("The shell scripts should now work properly in Linux/CoCalc.")
    else:
        print("All shell scripts already have Unix line endings.")

    # Also make shell scripts executable
    print()
    print("Making shell scripts executable...")
    for script_path in shell_scripts:
        try:
            # Add execute permission for owner
            os.chmod(script_path, os.stat(script_path).st_mode | 0o100)
            print(f"  ✓ Made {script_path.name} executable")
        except Exception as e:
            print(f"  ✗ Could not make {script_path.name} executable: {e}")


if __name__ == "__main__":
    main()