# DS776 Course Cleanup - Next Session Tasks

## Session Setup
1. **Start VS Code from WSL terminal** to inherit environment variables:
   ```bash
   cd /mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776
   code .
   ```
   This ensures `DS776_ROOT_DIR` is available in VS Code and Jupyter notebooks.

2. **Verify environment**:
   ```bash
   echo $DS776_ROOT_DIR
   # Should output: /mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776
   ```

## Remaining Tasks

### 1. Update config_paths_keys Function
**File**: `Lessons/Course_Tools/introdl/src/introdl/utils/utils.py`

- [ ] Update the `config_paths_keys()` function to use the new path utilities from `path_utils.py`
- [ ] Replace hardcoded paths like `~/Lessons/Course_Tools/` with calls to `resolve_env_file()`
- [ ] Use `get_workspace_dir()` for workspace paths instead of hardcoded defaults
- [ ] Test with both DS776_ROOT_DIR set (your environment) and unset (student environment)

### 2. Update Notebook Installation Cells
**Pattern to update in ALL lesson notebooks**:

Current pattern (found in many notebooks):
```python
# run this cell to ensure the course package is installed
import sys
from pathlib import Path

# This path assumes Lessons is in your main user folder like on CoCalc.
# Might need to change this in other environments
course_tools_path = Path('~/Lessons/Course_Tools').expanduser().resolve()
sys.path.append(str(course_tools_path))

# Import and run the installer function
from install_introdl import install_course_package
install_course_package()
```

Replace with:
```python
# Run this cell to ensure the course package is installed
# If running locally, set DS776_ROOT_DIR environment variable to your course root
import sys
from pathlib import Path
import os

# Find Course_Tools directory (checks DS776_ROOT_DIR if set)
if 'DS776_ROOT_DIR' in os.environ:
    course_tools_path = Path(os.environ['DS776_ROOT_DIR']) / 'Lessons' / 'Course_Tools'
else:
    course_tools_path = Path('~/Lessons/Course_Tools').expanduser()

sys.path.append(str(course_tools_path))

# Import and run the installer function
from install_introdl import install_course_package
install_course_package()
```

**Notebooks to update** (search for pattern in):
- All notebooks in `Lessons/Lesson_*/L*.ipynb`
- All notebooks in `Homework/HW*.ipynb`
- All notebooks in `Solutions/HW*/`

### 3. Update Install_and_Clean.ipynb
**File**: `Lessons/Course_Tools/Install_and_Clean.ipynb`

- [ ] Add section explaining DS776_ROOT_DIR for local development
- [ ] Update installation code to use the new flexible path system
- [ ] Add instructions for local developers vs. CoCalc students
- [ ] Test all cleaning functions with the new path structure
- [ ] Add a cell to detect and display the current environment configuration

Suggested new cell to add at the top:
```python
# Environment Detection and Configuration
import os
from pathlib import Path

print("🔍 Environment Configuration:")
print("-" * 40)

# Check for DS776_ROOT_DIR
if 'DS776_ROOT_DIR' in os.environ:
    root_dir = Path(os.environ['DS776_ROOT_DIR'])
    print(f"✅ DS776_ROOT_DIR is set: {root_dir}")
    print("   Running in local development mode")
else:
    root_dir = Path.home()
    print(f"📚 Using default paths from: {root_dir}")
    print("   Running in student mode (CoCalc/Colab)")

# Display expected paths
print("\n📁 Expected Course Structure:")
print(f"   Lessons:    {root_dir / 'Lessons'}")
print(f"   Homework:   {root_dir / 'Homework'}")
print(f"   Solutions:  {root_dir / 'Solutions'}")

# Check which paths exist
print("\n✓ Path Status:")
for dir_name in ['Lessons', 'Homework', 'Solutions']:
    path = root_dir / dir_name
    status = "✅ Exists" if path.exists() else "❌ Not found"
    print(f"   {dir_name}: {status}")
```

### 4. Testing Checklist
After making changes:

- [ ] Test package installation on Windows/WSL with DS776_ROOT_DIR set
- [ ] Test package installation simulation without DS776_ROOT_DIR (student scenario)
- [ ] Verify notebooks can find and import introdl package
- [ ] Test config_paths_keys() in both environments
- [ ] Run Install_and_Clean.ipynb completely
- [ ] Test at least one notebook from each lesson folder

### 5. Final Cleanup
- [ ] Remove any remaining hardcoded paths that assume `~/Lessons` structure
- [ ] Update CLAUDE.md if needed to document the DS776_ROOT_DIR usage
- [ ] Consider adding a setup script for local developers
- [ ] Merge feature branch back to main

## Notes from Previous Session
- Created `path_utils.py` module with flexible path resolution
- Updated `install_introdl.py` to check DS776_ROOT_DIR
- Added DS776_ROOT_DIR to ~/.bashrc
- Created feature branch: `feature/flexible-path-resolution`

## Branch Information
Currently on: `feature/flexible-path-resolution`
Main branch: `main`

Remember to test thoroughly before merging!