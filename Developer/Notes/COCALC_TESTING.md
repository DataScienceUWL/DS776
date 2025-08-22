# CoCalc Testing Guide for DS776

## Quick Setup Instructions

### 1. Copy Scripts to CoCalc
In your local terminal:
```bash
# Copy the test scripts to CoCalc
scp test_in_cocalc.sh update_cocalc_test.sh [your-cocalc-project]
```

### 2. Run Setup Script in CoCalc Terminal
```bash
# Make scripts executable
chmod +x test_in_cocalc.sh update_cocalc_test.sh

# Run the setup script
./test_in_cocalc.sh
```

This will:
- Clone the repository from feature/flexible-path-resolution branch
- Create symbolic links so folders appear as ~/Lessons, ~/Homework, etc.
- Set up the structure for testing

### 3. Test the Installation

#### In Install_and_Clean.ipynb:
1. **Run environment detection cell** - Should show:
   - "✅ Detected environment: CoCalc Home Server" (or Compute Server)
   - Should NOT show DS776_ROOT_DIR

2. **Run installation cell** - Should:
   - Install introdl in editable mode from local path
   - Show "📦 Installing introdl in editable mode"

#### In L02_1_MNIST_FC.ipynb:
1. **Run Cell 0 (package check)** - Should show:
   - "💡 Tip: Use Install_and_Clean.ipynb..." (always displays)
   - "✅ introdl package is installed"

2. **Run Cell 1 (imports and config)** - Should show:
   - Environment detection (CoCalc)
   - Creates and uses paths:
     - DATA_PATH=~/home_workspace/data
     - MODELS_PATH=~/home_workspace/models
     - CACHE_PATH=~/home_workspace/downloads
   - These directories should be created automatically

## Expected Behavior by Environment

### CoCalc Home Server
- Environment: "cocalc"
- Workspace: ~/home_workspace/
- Env file: cocalc.env (if exists)

### CoCalc Compute Server
- Environment: "cocalc_compute_server"
- Workspace: ~/cs_workspace/
- Env file: cocalc_compute_server.env (if exists)

### Your Local Dev (with DS776_ROOT_DIR)
- Environment: "unknown" (or "vscode")
- Workspace: $DS776_ROOT_DIR/data, models, downloads
- Skips env file loading

## Updating After Changes

When you push new changes:
```bash
cd ~/DS776_test
git pull origin feature/flexible-path-resolution

# If introdl package changed, restart kernel and reinstall
```

Or use the update script:
```bash
./update_cocalc_test.sh
```

## Troubleshooting

### If paths are wrong:
1. Check environment detection is correct
2. Verify no DS776_ROOT_DIR is set: `echo $DS776_ROOT_DIR`
3. Check if old env vars are cached: `echo $DATA_PATH`
4. Restart kernel to clear cached values

### If package not found:
1. Run installation cell in Install_and_Clean.ipynb
2. Restart kernel after installation
3. Check installation with: `pip show introdl`

### To completely reset:
```bash
# Remove everything
rm -rf ~/DS776_test ~/Lessons ~/Homework ~/Solutions ~/Textbooks
rm -rf ~/home_workspace  # or ~/cs_workspace on compute server

# Run setup again
./test_in_cocalc.sh
```

## What We're Testing

1. **Environment Detection** - Correctly identifies CoCalc vs local dev
2. **Path Resolution** - Uses appropriate directories for each environment
3. **Package Installation** - Editable mode installation works
4. **API Key Loading** - Loads from ~/api_keys.env if present
5. **Directory Creation** - Automatically creates workspace directories

## Success Criteria

✅ CoCalc detected correctly
✅ Paths set to ~/home_workspace/* (not ~/data, ~/models)
✅ Directories created automatically
✅ Package installs in editable mode
✅ No errors about missing directories
✅ Students don't need to manually create directories