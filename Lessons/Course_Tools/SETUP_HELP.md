# DS776 Setup Help

This guide helps troubleshoot common setup issues with the DS776 course materials.

## Quick Start

Every notebook starts with two required setup cells:

### Cell 1: Environment & Package Setup
```python
%run ../Course_Tools/auto_update_introdl.py  # (path varies by location)
```

**What it does:**
- Configures cache paths (HuggingFace, PyTorch) for proper storage management
- Checks if the `introdl` course package needs updating
- Updates automatically if a new version is available
- Suppresses TensorFlow/Keras warnings

### Cell 2: Imports & Path Configuration
```python
from introdl import config_paths_keys, ...

paths = config_paths_keys()
DATA_PATH = paths['DATA_PATH']
MODELS_PATH = paths['MODELS_PATH']
```

**What it does:**
- Loads the course utility functions
- Detects your environment (CoCalc, local, etc.)
- Sets up path variables for data, models, and cache
- Loads API keys from `api_keys.env`

---

## Common Issues and Solutions

### "File not found" or "%run failed"

**Cause:** The path to `auto_update_introdl.py` is incorrect.

**Solution:**
1. Make sure you have the complete course files
2. Check the relative path:
   - From `Lessons/Lesson_XX/`: use `../Course_Tools/auto_update_introdl.py`
   - From `Homework/Homework_XX/`: use `../../Lessons/Course_Tools/auto_update_introdl.py`

### "ModuleNotFoundError: No module named 'introdl'"

**Cause:** The introdl package isn't installed or the auto-update failed.

**Solution:**
1. Run the Course_Setup notebook: `Lessons/Course_Tools/Course_Setup.ipynb`
2. Restart your kernel after setup completes
3. Try your notebook again

### "KERNEL RESTART REQUIRED"

**Cause:** The introdl package was updated and Python needs to reload it.

**Solution:**
1. Restart your kernel (Kernel → Restart)
2. Run the first cell again
3. You should see "introdl vX.X.X ready" without the restart message

### Import Errors After Update

**Cause:** Old cached bytecode conflicts with new package version.

**Solution:**
1. Restart your kernel completely
2. If still failing, run `Course_Setup.ipynb` with the "force reinstall" option
3. Restart kernel and try again

### Storage Full / Disk Quota Exceeded

**Cause:** Cached models, checkpoints, or datasets filled your storage.

**Solution:**
1. Run `Lessons/Course_Tools/Storage_Cleanup.ipynb`
2. For HuggingFace Trainer, always use `save_total_limit=1` in TrainingArguments
3. Delete old homework model folders after grades are posted

---

## Understanding the Setup System

### Why Two Setup Cells?

**Cell 1 (`%run auto_update_introdl.py`)** must run BEFORE any imports because:
- HuggingFace and PyTorch lock their cache paths at import time
- Environment variables must be set before `import torch` or `import transformers`
- This ensures downloads go to the right folders for storage management

**Cell 2 (`config_paths_keys()`)** runs AFTER imports because:
- It returns Python path variables for use in your code
- It loads API keys from your `api_keys.env` file
- It prints environment detection info

### Path Variables Explained

| Variable | Purpose | Typical Location |
|----------|---------|------------------|
| `DATA_PATH` | Datasets (CIFAR, IMDB, etc.) | `home_workspace/data/` |
| `MODELS_PATH` | Your trained model checkpoints | `Lesson_XX_Models/` or `Homework_XX_Models/` |
| `CACHE_PATH` | Pretrained model downloads | `home_workspace/downloads/` |

**Always use these variables instead of hardcoding paths!**

```python
# Good:
torch.save(model.state_dict(), MODELS_PATH / 'my_model.pt')
dataset = load_dataset(DATA_PATH / 'cifar10')

# Bad - will break on different environments:
torch.save(model.state_dict(), './models/my_model.pt')
dataset = load_dataset('/home/user/data/cifar10')
```

---

## Environment-Specific Notes

### CoCalc (Home Server)
- Storage limit: ~10GB for synced files
- All workspace folders sync and count against limit
- Use `Storage_Cleanup.ipynb` regularly

### CoCalc (Compute Server)
- `home_workspace/` syncs back (~10GB limit)
- `cs_workspace/` is local only (~50GB, faster)
- Data and cache automatically use `cs_workspace/`
- Your models sync back via `MODELS_PATH`

### Local Development
- Set `DS776_ROOT_DIR` environment variable to your course folder
- Paths will be relative to that directory
- Run `git pull` to get course updates

### Google Colab
- Storage is temporary (cleared when runtime ends)
- Mount Google Drive for persistent storage
- See `Course_Setup.ipynb` for Colab-specific instructions

---

## Getting Help

If you're still having issues:

1. **Check Piazza** for similar questions and solutions
2. **Post on Piazza** with:
   - Your environment (CoCalc, local, Colab)
   - The exact error message
   - Which notebook you're running
   - Whether `Course_Setup.ipynb` worked

3. **Office Hours** - bring your laptop for hands-on help

---

## Technical Reference

### Auto-Update Script Behavior

| Exit Code | Meaning | Action Needed |
|-----------|---------|---------------|
| 0 | Success, ready to proceed | None |
| 1 | Error occurred | Run Course_Setup.ipynb |
| 2 | Package updated | Restart kernel, re-run cell |

### Version Checking

The auto-update script compares:
1. **Installed version**: Currently installed `introdl` package
2. **Source version**: Version in `Course_Tools/introdl/src/introdl/__init__.py`

Updates happen only when versions differ.

### Files in Course_Tools

| File | Purpose |
|------|---------|
| `auto_update_introdl.py` | Main setup/update script |
| `Course_Setup.ipynb` | Full setup notebook |
| `Storage_Cleanup.ipynb` | Free up storage space |
| `api_keys.env` | Template for API keys |
| `introdl/` | Course package source |
