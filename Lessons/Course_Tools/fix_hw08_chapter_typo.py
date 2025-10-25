#!/usr/bin/env python3
"""
Surgical fix for Homework 08 Chapter typo.

Fixes "based on Chapter 3: Text Classification" → "based on Chapter 2: Text Classification"
in Homework_08_Assignment*.ipynb files.

This script:
1. Finds all Homework_08_Assignment*.ipynb files
2. Locates the specific typo in the Reading Questions section
3. Replaces only that specific text
4. Preserves all other notebook content and structure

Usage:
    python fix_hw08_chapter_typo.py
"""

import json
from pathlib import Path


def fix_notebook(notebook_path: Path) -> bool:
    """
    Fix the chapter typo in a single notebook.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        True if changes were made, False otherwise
    """
    print(f"\nProcessing: {notebook_path}")

    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Track if we made changes
    changes_made = False

    # Search for the typo in markdown cells
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            # Get the cell source (can be string or list of strings)
            source = cell.get('source', '')

            # Convert to string if it's a list
            if isinstance(source, list):
                source_str = ''.join(source)
            else:
                source_str = source

            # Check if this cell contains the typo
            if 'based on Chapter 3: Text Classification' in source_str:
                # Fix the typo
                if isinstance(source, list):
                    # Fix in list format
                    fixed_source = []
                    for line in source:
                        fixed_line = line.replace(
                            'based on Chapter 3: Text Classification',
                            'based on Chapter 2: Text Classification'
                        )
                        fixed_source.append(fixed_line)
                    cell['source'] = fixed_source
                else:
                    # Fix in string format
                    cell['source'] = source_str.replace(
                        'based on Chapter 3: Text Classification',
                        'based on Chapter 2: Text Classification'
                    )

                changes_made = True
                print(f"  ✓ Fixed typo: Chapter 3 → Chapter 2")

    if changes_made:
        # Save the fixed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
            f.write('\n')  # Add trailing newline
        print(f"  ✓ Saved changes to: {notebook_path.name}")
        return True
    else:
        print(f"  ℹ No changes needed (typo not found)")
        return False


def main():
    """Main function to find and fix all matching notebooks."""
    print("=" * 70)
    print("Homework 08 Chapter Typo Fix")
    print("=" * 70)
    print("\nSearching for Homework_08_Assignment*.ipynb files...")

    # Find the Homework_08 directory
    # Script is in Lessons/Course_Tools, so homework is ../../Homework/Homework_08
    script_dir = Path(__file__).parent
    hw08_dir = script_dir / '..' / '..' / 'Homework' / 'Homework_08'
    hw08_dir = hw08_dir.resolve()

    if not hw08_dir.exists():
        print(f"\n❌ ERROR: Homework_08 directory not found at: {hw08_dir}")
        print("\nPlease run this script from the course repository.")
        return 1

    # Find all matching notebooks
    notebooks = list(hw08_dir.glob('Homework_08_Assignment*.ipynb'))

    if not notebooks:
        print(f"\n❌ No matching notebooks found in: {hw08_dir}")
        return 1

    print(f"\nFound {len(notebooks)} notebook(s):")
    for nb in notebooks:
        print(f"  - {nb.name}")

    # Process each notebook
    total_fixed = 0
    for notebook_path in notebooks:
        if fix_notebook(notebook_path):
            total_fixed += 1

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Notebooks processed: {len(notebooks)}")
    print(f"Notebooks modified:  {total_fixed}")
    print(f"Notebooks unchanged: {len(notebooks) - total_fixed}")

    if total_fixed > 0:
        print("\n✓ Fix applied successfully!")
    else:
        print("\nℹ No modifications were necessary.")

    return 0


if __name__ == '__main__':
    exit(main())
