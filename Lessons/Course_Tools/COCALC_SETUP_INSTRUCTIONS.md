# CoCalc Setup and Troubleshooting Instructions

**For Claude Code or instructor use in CoCalc**

This file contains instructions for diagnosing and fixing environment setup issues in CoCalc.

---

## Current Issue

The environment detection in `auto_update_introdl.py` is not correctly identifying CoCalc. It should detect:
- `~/.cocalc` directory exists → CoCalc environment
- `~/cs_workspace` exists → Compute Server (use cs_workspace for cache)
- Only `~/home_workspace` exists → Home Server (use home_workspace for cache)

---

## Diagnostic Steps

### Step 1: Check CoCalc indicators

Run in terminal or Python:
```python
from pathlib import Path
home = Path.home()

print(f"Home: {home}")
print(f"~/.cocalc exists: {(home / '.cocalc').exists()}")
print(f"~/.smc exists: {(home / '.smc').exists()}")
print(f"~/home_workspace exists: {(home / 'home_workspace').exists()}")
print(f"~/cs_workspace exists: {(home / 'cs_workspace').exists()}")
```

### Step 2: Check environment variables

```python
import os
print(f"COCALC_PROJECT_ID: {os.environ.get('COCALC_PROJECT_ID', 'NOT SET')}")
print(f"SMC: {os.environ.get('SMC', 'NOT SET')}")
print(f"TORCH_HOME: {os.environ.get('TORCH_HOME', 'NOT SET')}")
print(f"HF_HOME: {os.environ.get('HF_HOME', 'NOT SET')}")
```

### Step 3: Check what auto_update sees

After running the auto_update cell, check:
```python
import os
print("Cache-related environment variables:")
for var in ['TORCH_HOME', 'HF_HOME', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME']:
    print(f"  {var}: {os.environ.get(var, 'NOT SET')}")
```

---

## Fixes

### If ~/.cocalc doesn't exist but this IS CoCalc

The environment detection needs to also check for `COCALC_PROJECT_ID` or other indicators.

**Fix in `auto_update_introdl.py`** around line 50:

Change:
```python
if (home / '.cocalc').exists():
```

To:
```python
if (home / '.cocalc').exists() or (home / '.smc').exists() or 'COCALC_PROJECT_ID' in os.environ:
```

**Also fix in `__init__.py`** around line 32:

Change:
```python
if (_home / '.cocalc').exists():
```

To:
```python
if (_home / '.cocalc').exists() or (_home / '.smc').exists() or 'COCALC_PROJECT_ID' in os.environ:
```

### If workspace directories don't exist

Create them:
```bash
mkdir -p ~/home_workspace/downloads
mkdir -p ~/home_workspace/data
mkdir -p ~/home_workspace/models
```

For compute server also:
```bash
mkdir -p ~/cs_workspace/downloads
mkdir -p ~/cs_workspace/data
```

---

## Updating the Repository

### If there are merge conflicts with untracked files

```bash
cd ~/DS776_repo
# Remove conflicting untracked files
rm -f Homework/Homework_12/Homework_12_Colab_Version.ipynb
rm -f Lessons/Lesson_13_Project/L13_1_Project_Choice_and_Draft.ipynb
# Pull latest
git pull origin main
```

### Sync to working directories

```bash
cd ~
# Remove old and copy fresh
rm -rf ~/Lessons ~/Homework
cp -r ~/DS776_repo/Lessons ~/Lessons
cp -r ~/DS776_repo/Homework ~/Homework
```

### Reinstall introdl package

```bash
pip uninstall introdl -y
cd ~/Lessons/Course_Tools/introdl
pip install . --no-cache-dir
```

### Verify installation

```bash
python -c "import introdl; print(f'introdl v{introdl.__version__}')"
```

---

## Testing After Fixes

Run the test notebooks in `Homework/Homework_Tests/`:

1. **Test_01_Environment_Variables.ipynb** - Should show:
   - Environment: "CoCalc" or "CoCalc COMPUTE SERVER"
   - All cache paths pointing to home_workspace or cs_workspace
   - NOT ~/.cache

2. **Test_05_Full_Verification.ipynb** - Should show:
   - All tests PASS
   - Files in correct locations

---

## Expected Behavior

### On CoCalc Home Server:
```
Environment: CoCalc
TORCH_HOME: /home/user/home_workspace/downloads
HF_HOME: /home/user/home_workspace/downloads/huggingface
HF_DATASETS_CACHE: /home/user/home_workspace/data
```

### On Compute Server:
```
Environment: CoCalc COMPUTE SERVER
TORCH_HOME: /home/user/cs_workspace/downloads
HF_HOME: /home/user/cs_workspace/downloads/huggingface
HF_DATASETS_CACHE: /home/user/cs_workspace/data
```

---

## File Locations Reference

| File | Purpose |
|------|---------|
| `Lessons/Course_Tools/auto_update_introdl.py` | Early env setup + package update |
| `Lessons/Course_Tools/introdl/src/introdl/__init__.py` | Backup env setup + package code |
| `Lessons/Course_Tools/introdl/src/introdl/utils.py` | config_paths_keys() function |
| `Homework/Homework_Tests/` | Test notebooks for verification |
