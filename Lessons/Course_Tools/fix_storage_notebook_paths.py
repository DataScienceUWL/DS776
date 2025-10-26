#!/usr/bin/env python3
"""
Fix auto-update paths in storage notebooks after distribution.

This script ensures homework notebooks use the correct relative path
to auto_update_introdl.py
"""
import json
import sys
from pathlib import Path

def fix_notebook_path(notebook_path):
    """Fix the auto-update path in a notebook."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Check if first cell is auto-update
    if len(nb['cells']) == 0:
        return False

    first_cell = nb['cells'][0]
    if first_cell['cell_type'] != 'code':
        return False

    source = ''.join(first_cell['source'])
    if 'Auto-Update' not in source or '%run' not in source:
        return False

    # Fix the path
    modified = False
    new_source = []
    for line in first_cell['source']:
        if '%run' in line and './auto_update_introdl.py' in line:
            new_line = line.replace('./auto_update_introdl.py',
                                   '../../Lessons/Course_Tools/auto_update_introdl.py')
            new_source.append(new_line)
            modified = True
        else:
            new_source.append(line)

    if modified:
        first_cell['source'] = new_source
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        return True

    return False

if __name__ == '__main__':
    # Get DS776 root (two levels up from this script)
    script_dir = Path(__file__).parent
    ds776_root = script_dir.parent.parent
    homework_dir = ds776_root / 'Homework'

    print("Fixing auto-update paths in storage notebooks...")
    print(f"Homework directory: {homework_dir}")
    print()

    fixed_count = 0
    for hw_folder in sorted(homework_dir.glob('Homework_*')):
        if hw_folder.is_dir():
            notebook = hw_folder / 'Storage_Cleanup_After_HW.ipynb'
            if notebook.exists():
                if fix_notebook_path(notebook):
                    print(f"✅ Fixed: {hw_folder.name}/Storage_Cleanup_After_HW.ipynb")
                    fixed_count += 1
                else:
                    print(f"⏭️  Skipped: {hw_folder.name}/Storage_Cleanup_After_HW.ipynb (already correct or no auto-update)")

    print()
    print(f"Total fixed: {fixed_count}")
