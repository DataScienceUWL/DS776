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

## Completed Tasks ✅
- [x] Updated config_paths_keys() to use path_utils.py
- [x] Fixed environment variable override when DS776_ROOT_DIR is set
- [x] Created flexible installation system with DS776_ROOT_DIR support
- [x] Fixed warning suppressions (tqdm, HuggingFace)
- [x] Implemented comprehensive model cache management
- [x] Fixed PyTorch gradient warning
- [x] Created CoCalc testing scripts (copy folders, not symlinks)
- [x] Updated L02_1_MNIST_FC.ipynb with new installation approach

## Remaining Tasks

### 1. Add Version Tracking to introdl Package
**File**: `Lessons/Course_Tools/introdl/src/introdl/__init__.py`

- [ ] Add __version__ attribute to package
- [ ] Display version in config_paths_keys() output
- [ ] Create version update workflow (bump version on each update)
- [ ] Students can see version without reinstalling (due to -e mode)

### 2. Test Model Caching and Cleanup
**Test notebooks from different lessons to verify cache management**

#### Lesson 2 (L02_1_MNIST_FC.ipynb) - Basic PyTorch models
- [ ] Run notebook and verify models save to MODELS_PATH
- [ ] Check torchvision datasets go to DATA_PATH
- [ ] Test cleanup with check_cache_usage() and clear_model_cache()

#### Lesson 5 (Transfer Learning) - Pretrained torchvision models
- [ ] Download pretrained models (ResNet, etc.)
- [ ] Verify they cache to CACHE_PATH/hub/
- [ ] Test that TORCH_HOME and TORCH_HUB env vars work
- [ ] Clean cache and verify re-download works

#### Lesson 7 (Transformers) - HuggingFace models
- [ ] Download transformer models
- [ ] Verify they cache to CACHE_PATH/huggingface/
- [ ] Test that HF_HOME, TRANSFORMERS_CACHE env vars work
- [ ] Clean cache and verify re-download works

### 3. Update Install_and_Clean.ipynb with Cache Management
**File**: `Lessons/Course_Tools/Install_and_Clean.ipynb`

- [ ] Add new cache management section using check_cache_usage()
- [ ] Add selective cleanup options using clear_model_cache()
- [ ] Show cache directory structure and explain separation
- [ ] Add cells for monitoring disk usage by cache type
- [ ] Include warnings about what's safe to delete when

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