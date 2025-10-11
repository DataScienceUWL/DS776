#!/usr/bin/env python3
"""
Cleanup script for student homework folders.

This script performs the following tasks:
1. Remove all *Utilities*.ipynb notebooks from Homework folders
2. Copy Storage_Cleanup_After_HW.ipynb from Course_Tools to Homework_05 through Homework_12
3. Remove all backup files ending with ~ from Homework_07 through Homework_12

Usage:
    python cleanup_student_homework.py

The script will automatically find the DS776 course root directory and perform
all cleanup operations.
"""

import os
import shutil
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

        # Summary
        print("=" * 70)
        print("CLEANUP COMPLETE")
        print("=" * 70)
        print(f"  Utilities notebooks removed:  {len(removed_utilities)}")
        print(f"  Storage_Cleanup files copied: {len(copied_cleanup)}")
        print(f"  Backup files removed:         {len(removed_backups)}")
        print()

        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
