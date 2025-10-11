#!/usr/bin/env python3
"""
Cleanup script for student homework folders.

This script performs the following tasks:
1. Remove all *Utilities*.ipynb notebooks from Homework folders
2. Copy Storage_Cleanup_After_HW.ipynb from Course_Tools to Homework_05 through Homework_12
3. Remove all backup files ending with ~ from Homework_07 through Homework_12
4. Remove old Homework_07_Storage* files from Homework_07 folder
5. Fix import order in Homework_07_Assignment.ipynb (introdl before transformers)

Usage:
    python cleanup_student_homework.py

The script will automatically find the DS776 course root directory and perform
all cleanup operations.
"""

import os
import shutil
import json
from pathlib import Path


def find_course_root():
    """Find the DS776 course root directory."""
    # Try common locations
    possible_roots = [
        Path.home() / "DS776",
        Path.home() / "DS776_new",
        Path.cwd()
    ]

    for root in possible_roots:
        if root.exists() and (root / "Lessons" / "Course_Tools").exists():
            return root

    # If not found, assume we're running from Course_Tools
    current = Path.cwd()
    if current.name == "Course_Tools":
        return current.parent.parent

    raise FileNotFoundError(
        "Could not find DS776 course root directory. "
        "Please run this script from within the DS776 directory structure."
    )


def remove_utilities_notebooks(homework_dir):
    """Remove all *Utilities*.ipynb files from homework folders."""
    removed_files = []

    for hw_folder in homework_dir.glob("Homework_*"):
        if hw_folder.is_dir():
            for utilities_file in hw_folder.glob("*Utilities*.ipynb"):
                print(f"  Removing: {utilities_file.relative_to(homework_dir)}")
                utilities_file.unlink()
                removed_files.append(utilities_file)

    return removed_files


def copy_storage_cleanup(course_root, homework_dir):
    """Copy Storage_Cleanup_After_HW.ipynb to Homework_05 through Homework_12."""
    source = course_root / "Lessons" / "Course_Tools" / "Storage_Cleanup_After_HW.ipynb"

    if not source.exists():
        print(f"  ⚠️  Warning: Storage_Cleanup_After_HW.ipynb not found in Course_Tools")
        return []

    copied_files = []

    for hw_num in range(5, 13):  # 5 through 12
        hw_folder = homework_dir / f"Homework_{hw_num:02d}"
        if hw_folder.exists() and hw_folder.is_dir():
            dest = hw_folder / "Storage_Cleanup_After_HW.ipynb"
            print(f"  Copying Storage_Cleanup to: {dest.relative_to(homework_dir)}")
            shutil.copy2(source, dest)
            copied_files.append(dest)
        else:
            print(f"  ⚠️  Skipping: {hw_folder.name} (folder not found)")

    return copied_files


def remove_backup_files(homework_dir):
    """Remove all files ending with ~ from Homework_07 through Homework_12."""
    removed_files = []

    for hw_num in range(7, 13):  # 7 through 12
        hw_folder = homework_dir / f"Homework_{hw_num:02d}"
        if hw_folder.exists() and hw_folder.is_dir():
            for backup_file in hw_folder.glob("*~"):
                print(f"  Removing backup: {backup_file.relative_to(homework_dir)}")
                backup_file.unlink()
                removed_files.append(backup_file)

    return removed_files


def remove_old_storage_notebooks(homework_dir):
    """Remove old Homework_07_Storage* files from Homework_07 folder."""
    removed_files = []

    hw_folder = homework_dir / "Homework_07"
    if hw_folder.exists() and hw_folder.is_dir():
        for storage_file in hw_folder.glob("Homework_07_Storage*"):
            print(f"  Removing old storage file: {storage_file.relative_to(homework_dir)}")
            storage_file.unlink()
            removed_files.append(storage_file)

    return removed_files


def fix_homework_07_imports(homework_dir):
    """
    Fix import order in Homework_07_Assignment.ipynb.

    Ensures introdl is imported before transformers to prevent Keras 3 compatibility issues.
    Only modifies the import cell if it has the wrong order.
    """
    hw_07_notebook = homework_dir / "Homework_07" / "Homework_07_Assignment.ipynb"

    if not hw_07_notebook.exists():
        print(f"  ⚠️  Skipping: Homework_07_Assignment.ipynb not found")
        return False

    try:
        # Read the notebook
        with open(hw_07_notebook, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find the imports cell (should be cell index 2 based on structure)
        modified = False
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

                # Check if this is the imports cell (has both transformers and introdl imports)
                if 'from transformers import' in source and 'from introdl import' in source:
                    # Check if transformers comes before introdl (wrong order)
                    transformers_pos = source.find('from transformers import')
                    introdl_pos = source.find('from introdl import')

                    if transformers_pos < introdl_pos and transformers_pos != -1:
                        print(f"  Found incorrect import order in Homework_07_Assignment.ipynb")

                        # Rewrite the cell with correct order
                        new_source = """import os
import torch
from introdl import (
    get_device,
    wrap_print_text,
    config_paths_keys,
    llm_generate,
    clear_pipeline,
    print_pipeline_info,
    display_markdown,
    show_session_spending
)
from transformers import pipeline
# Wrap print to format text nicely at 120 characters
print = wrap_print_text(print, width=120)

device = get_device()

paths = config_paths_keys()
"""
                        cell['source'] = new_source
                        modified = True
                        break

        if modified:
            # Write back the notebook
            with open(hw_07_notebook, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)

            print(f"  ✓ Fixed import order: introdl now imported before transformers")
            return True
        else:
            print(f"  ✓ Import order already correct (introdl before transformers)")
            return False

    except Exception as e:
        print(f"  ⚠️  Error fixing imports: {e}")
        return False


def main():
    """Main cleanup function."""
    print("=" * 70)
    print("DS776 Student Homework Cleanup Script")
    print("=" * 70)
    print()

    try:
        # Find course root
        course_root = find_course_root()
        print(f"✓ Found course root: {course_root}")
        print()

        homework_dir = course_root / "Homework"
        if not homework_dir.exists():
            print(f"✗ Error: Homework directory not found at {homework_dir}")
            return 1

        # Task 1: Remove Utilities notebooks
        print("Task 1: Removing *Utilities*.ipynb files...")
        removed_utilities = remove_utilities_notebooks(homework_dir)
        print(f"✓ Removed {len(removed_utilities)} Utilities notebooks")
        print()

        # Task 2: Copy Storage_Cleanup notebooks
        print("Task 2: Copying Storage_Cleanup_After_HW.ipynb to Homework_05-12...")
        copied_cleanup = copy_storage_cleanup(course_root, homework_dir)
        print(f"✓ Copied Storage_Cleanup to {len(copied_cleanup)} homework folders")
        print()

        # Task 3: Remove backup files
        print("Task 3: Removing backup files (*~) from Homework_07-12...")
        removed_backups = remove_backup_files(homework_dir)
        print(f"✓ Removed {len(removed_backups)} backup files")
        print()

        # Task 4: Remove old storage notebooks from Homework_07
        print("Task 4: Removing old Homework_07_Storage* files from Homework_07...")
        removed_storage = remove_old_storage_notebooks(homework_dir)
        print(f"✓ Removed {len(removed_storage)} old storage files")
        print()

        # Task 5: Fix import order in Homework_07
        print("Task 5: Fixing import order in Homework_07_Assignment.ipynb...")
        fixed_imports = fix_homework_07_imports(homework_dir)
        print(f"✓ Import order check complete")
        print()

        # Summary
        print("=" * 70)
        print("CLEANUP COMPLETE")
        print("=" * 70)
        print(f"  Utilities notebooks removed:  {len(removed_utilities)}")
        print(f"  Storage_Cleanup files copied: {len(copied_cleanup)}")
        print(f"  Backup files removed:         {len(removed_backups)}")
        print(f"  Old storage files removed:    {len(removed_storage)}")
        print(f"  Homework_07 imports fixed:    {'Yes' if fixed_imports else 'Already correct'}")
        print()

        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
