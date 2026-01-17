# Homework Tests

Test notebooks for verifying the DS776 environment setup and cache path configuration.

## Purpose

These tests verify that:
1. Environment variables (TORCH_HOME, HF_HOME, etc.) are set correctly
2. Import order doesn't affect cache path configuration
3. HuggingFace downloads go to the correct location
4. Torchvision/PyTorch downloads go to the correct location
5. Storage_Cleanup.ipynb will find files in the correct locations

## Test Notebooks

| Notebook | Purpose |
|----------|---------|
| Test_01_Environment_Variables.ipynb | Verify env vars are set before imports |
| Test_02_Import_Order.ipynb | Verify import order doesn't matter |
| Test_03_HuggingFace_Cache.ipynb | Download HF models/datasets and verify location |
| Test_04_Torchvision_Cache.ipynb | Download torch models and verify location |
| Test_05_Full_Verification.ipynb | Complete storage audit and cleanup compatibility |

## How to Run

1. **Restart kernel** before each test (important for accurate results)
2. Run tests in order (01 → 05)
3. Run on both **CoCalc Home Server** and **Compute Server**

## Expected Results

**On CoCalc Home Server:**
- Downloads should go to `~/home_workspace/downloads/`
- Datasets should go to `~/home_workspace/data/`

**On Compute Server:**
- Downloads should go to `~/cs_workspace/downloads/`
- Datasets should go to `~/cs_workspace/data/`

**Never:**
- `~/.cache/huggingface/`
- `~/.cache/torch/`

## Troubleshooting

If tests fail:
1. Make sure you ran the auto_update cell (cell 0) first
2. Restart kernel and try again
3. Check if pre-existing files in `~/.cache` are causing issues
4. Use Test_05's optional cleanup cell to remove bad cache locations
