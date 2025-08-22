#!/usr/bin/env python3
"""
Generate Homework_XX_Utilities.ipynb notebooks from the HW_Utilities_Template.ipynb template.

This script:
1. Reads the template notebook
2. Creates a customized utility notebook for each homework folder
3. Updates placeholders with the correct homework number
4. Uses consistent naming: Homework_XX_Utilities.ipynb
"""

import json
import nbformat
from pathlib import Path
import sys

def create_hw_util_notebook(hw_num, template_path, output_dir):
    """
    Creates a homework utility notebook from a template.

    Args:
        hw_num (str): The homework number string (e.g., "01", "02").
        template_path (Path): The path to the HW_Utilities_Template.ipynb file.
        output_dir (Path): The directory to save the new notebook.
    """
    # Consistent naming pattern
    output_filename = f"Homework_{hw_num}_Utilities.ipynb"
    output_path = output_dir / output_filename

    # Read the template notebook
    with open(template_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Replace placeholders throughout the notebook
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and 'source' in cell:
            # Replace XX placeholders with actual homework number
            cell.source = cell.source.replace('Homework XX', f'Homework {hw_num}')
            cell.source = cell.source.replace('Lesson_XX', f'Lesson_{hw_num}')
            cell.source = cell.source.replace('Homework_XX', f'Homework_{hw_num}')
            
        elif cell.cell_type == 'code' and 'source' in cell:
            # Update code cell placeholders
            cell.source = cell.source.replace('"Homework_XX.ipynb"', f'"Homework_{hw_num}.ipynb"')
            cell.source = cell.source.replace("'XX'", f"'{hw_num}'")

    # Write the new notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"✅ Created: {output_path.name}")
    return output_path

def remove_old_utility_notebooks(hw_dir, hw_num):
    """Remove old utility notebooks with inconsistent naming."""
    old_patterns = [
        f"HW{hw_num}_Utilities.ipynb",
        f"HW{hw_num.lstrip('0')}_Utilities.ipynb",  # e.g., HW1_Utilities.ipynb
        f"HW{hw_num}_Utils.ipynb",
    ]
    
    removed = []
    for pattern in old_patterns:
        old_file = hw_dir / pattern
        if old_file.exists():
            old_file.unlink()
            removed.append(old_file.name)
            
    if removed:
        print(f"   🗑️  Removed old files: {', '.join(removed)}")

def create_all_hw_util_notebooks(template_path, base_output_dir):
    """
    Creates all homework utility notebooks for the semester.

    Args:
        template_path (Path): The path to the HW_Utilities_Template.ipynb file.
        base_output_dir (Path): The Homework directory containing all HW folders.
    """
    if not template_path.exists():
        print(f"❌ Error: Template notebook not found at {template_path}")
        return False

    print(f"📚 Generating homework utility notebooks from template...")
    print(f"   Template: {template_path.name}")
    print(f"   Output pattern: Homework_XX_Utilities.ipynb\n")

    created_count = 0
    
    # Find all Homework_XX directories
    hw_dirs = sorted([d for d in base_output_dir.glob("Homework_*") if d.is_dir()])
    
    if not hw_dirs:
        print(f"❌ No homework directories found in {base_output_dir}")
        return False
    
    for hw_dir in hw_dirs:
        # Extract homework number from directory name
        dir_name = hw_dir.name
        if dir_name.startswith("Homework_"):
            hw_num = dir_name.split("_")[1]  # Get the XX part
            
            print(f"📁 Processing {dir_name}:")
            
            # Remove old utility notebooks with inconsistent naming
            remove_old_utility_notebooks(hw_dir, hw_num)
            
            # Create new utility notebook
            output_path = create_hw_util_notebook(hw_num, template_path, hw_dir)
            created_count += 1
    
    print(f"\n✨ Successfully created {created_count} utility notebooks!")
    return True

def main():
    """Main function to generate all homework utility notebooks."""
    # Get paths - script is now in Developer/Scripts
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up to project root
    template_path = script_dir / "HW_Utilities_Template.ipynb"
    base_output_dir = project_root / "Homework"  # The Homework directory
    
    # Generate all notebooks
    success = create_all_hw_util_notebooks(template_path, base_output_dir)
    
    if success:
        print("\n📝 Next steps:")
        print("   1. Review the generated notebooks")
        print("   2. Test functionality in one notebook")
        print("   3. Commit changes to repository")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()