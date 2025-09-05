# DS776 Auto-Update System

This system provides seamless, automatic introdl package management for students.

## 🚀 **How It Works**

1. **Smart Version Checking** - Compares installed vs source version
2. **Fast Execution** - Usually <2 seconds when no update needed  
3. **Automatic Updates** - Only when instructor releases new versions
4. **Cross-Environment** - Works in CoCalc, local installations, etc.
5. **Health Checks** - Ensures API keys, workspace, etc. are configured

## 📋 **For Instructors**

### Adding to Existing Notebooks

Copy the content from `notebook_auto_update_cell.py` and paste it as the **first cell** in every lesson and homework notebook.

### Releasing Updates

1. **Update the version** in `introdl/src/introdl/__init__.py`:
   ```python
   __version__ = "1.4.2"  # Bump this version on each update
   ```

2. **Distribute updated files** to students (via GitHub, CoCalc sync, etc.)

3. **Students automatically get updates** when they run any notebook

### System Files

- `auto_update_introdl.sh` - The main auto-update script
- `notebook_auto_update_cell.py` - Template cell for notebooks
- This documentation file

## 📱 **For Students**

### Normal Usage

Just run your lesson/homework notebooks normally! The first cell automatically:
- ✅ Checks if introdl is up to date
- ✅ Updates only when needed  
- ✅ Runs health checks
- ✅ Takes care of setup automatically

### Local Environment Students

If you've cloned the GitHub repository locally:

1. **Set `DS776_ROOT_DIR`** to your local repo directory
2. **Run notebooks normally** - the auto-update cell works the same way
3. **Get update notifications** - You'll see when GitHub has newer versions
4. **Easy updates** - Just run `git pull origin main` when prompted
5. **Git status awareness** - See if your local repo is behind remote

### If Updates Are Needed

When the instructor releases updates, you'll see:
```
🔄 KERNEL RESTART REQUIRED
The introdl package was updated.
Please RESTART THE KERNEL and run this cell again.
```

### Local Repository with Updates Available
```
🔍 DS776 introdl Auto-Check
==========================
📦 Source version: 1.4.1
✅ Installed version: 1.4.1
ℹ️  Detected local git repository
✅ Version 1.4.1 is current - no update needed

🔧 Quick health check...
✅ introdl.utils imports correctly
✅ API keys file exists

⚠️  Newer version (1.4.2) available on GitHub!
ℹ️  To update: git pull origin main
ℹ️  Your current version: 1.4.1

⚠️  Your local repository is behind the remote!
ℹ️  To get latest course materials: git pull origin main
ℹ️  You are 3 commits behind

✅ All checks complete - ready to proceed!
======================================

✅ Ready to proceed with the lesson/homework!
```

Just restart your kernel and re-run the cell - that's it!

### Troubleshooting

If the auto-update cell fails or shows errors:

**Step 1: Try the Full Setup**
1. Navigate to `Lessons/Course_Tools/`
2. Open and run `Course_Setup.ipynb`
3. Restart your kernel when prompted
4. Return to your lesson/homework notebook

**Step 2: Manual Terminal Setup (if comfortable)**
1. Open a terminal
2. Run: `cd Lessons/Course_Tools`
3. Run: `bash setup_course.sh`
4. Restart your Python kernel

**Step 3: Check Common Issues**
- **"File not found"** → Make sure you have the complete course files
- **"bash command not found"** (Windows) → Use `Course_Setup.ipynb` instead
- **"Permission denied"** → You may need to run setup as administrator/root
- **Import errors** → Try restarting your kernel first

**Step 4: Get Help**
- Post on Piazza with:
  - Your operating system (Windows/Mac/Linux)
  - Environment (CoCalc/local/Colab)  
  - The exact error message from the auto-update cell
  - Whether `Course_Setup.ipynb` worked or not

## 🔧 **Technical Details**

### Script Behavior

- **Exit Code 0**: Everything OK, proceed with notebook
- **Exit Code 1**: Error occurred, may need manual setup  
- **Exit Code 2**: Update performed, restart kernel required

### Version Comparison

The script compares:
1. **Installed version** - `python3 -c "import introdl; print(introdl.__version__)"`
2. **Source version** - From `introdl/src/introdl/__init__.py`
3. **Updates only if different** or if path conflicts detected

### Environment Detection

Works automatically in:
- ✅ **CoCalc** (compute servers and main project)
- ✅ **Google Colab**  
- ✅ **Local Git Repository** (students with `DS776_ROOT_DIR` set)
- ✅ **Local Jupyter** installations
- ✅ **VS Code** with Jupyter extension

### Local Repository Features

For students with local git repositories, the system provides extra features:
- 🔍 **GitHub version checking** - Compares local vs latest GitHub version
- 📊 **Git status checking** - Shows if local repo is behind remote
- 💡 **Update instructions** - Clear `git pull` guidance when updates available
- 🏠 **Local environment detection** - Recognizes git repos vs cloud environments

### Paths Handled

- **From Lessons**: `../Course_Tools/auto_update_introdl.sh`
- **From Homework**: `../../Lessons/Course_Tools/auto_update_introdl.sh`  
- **Auto-detection**: Searches upward if standard paths don't work

## 🎯 **Benefits**

### For Students
- 🚀 **Seamless experience** - Just run notebooks, everything works
- ⚡ **Fast startup** - No waiting unless updates needed
- 🔄 **Always current** - Automatic updates from instructor
- 🛠️ **Self-healing** - Fixes common setup issues automatically

### For Instructors  
- 📦 **Easy updates** - Just bump version and distribute
- 📈 **Reduced support** - Fewer "setup doesn't work" questions
- 🔍 **Visibility** - Clear messaging about what's happening
- 🎛️ **Control** - Granular version management

## 📝 **Example Cell Output**

### Normal Run (No Updates Needed)
```
🔍 DS776 introdl Auto-Check
==========================
📦 Source version: 1.4.1
✅ Installed version: 1.4.1
ℹ️  Location: /usr/local/lib/python3.10/dist-packages/introdl/__init__.py
✅ Version 1.4.1 is current - no update needed

🔧 Quick health check...
✅ introdl.utils imports correctly
✅ API keys file exists
✅ All checks complete - ready to proceed!
======================================

✅ Ready to proceed with the lesson/homework!
```

### Update Required
```
🔍 DS776 introdl Auto-Check
==========================
📦 Source version: 1.4.2
✅ Installed version: 1.4.1
ℹ️  Version mismatch (1.4.1 != 1.4.2) - will update

📦 Installing/updating introdl...
================================
✅ Installation successful!
✅ Installation verified - introdl.utils imports correctly

🔄 IMPORTANT: Restart your kernel and run this cell again!
=========================================================

🔄 KERNEL RESTART REQUIRED
The introdl package was updated.
Please RESTART THE KERNEL and run this cell again.
```

### Local Repository with Updates Available
```
🔍 DS776 introdl Auto-Check
==========================
📦 Source version: 1.4.1
✅ Installed version: 1.4.1
ℹ️  Detected local git repository
✅ Version 1.4.1 is current - no update needed

🔧 Quick health check...
✅ introdl.utils imports correctly
✅ API keys file exists

⚠️  Newer version (1.4.2) available on GitHub!
ℹ️  To update: git pull origin main
ℹ️  Your current version: 1.4.1

⚠️  Your local repository is behind the remote!
ℹ️  To get latest course materials: git pull origin main
ℹ️  You are 3 commits behind

✅ All checks complete - ready to proceed!
======================================

✅ Ready to proceed with the lesson/homework!
```