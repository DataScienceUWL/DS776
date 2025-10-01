import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms.v2 as transforms
import sys
import os
import random
import numpy as np
import pandas as pd
import inspect
from torchinfo import summary
import traceback
from textwrap import TextWrapper
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pathlib import Path
import warnings
import shutil
from IPython.display import display, IFrame
from IPython.core.display import HTML
import gc
import nbformat
import nbformat
from nbformat import validate
from nbformat.validator import NotebookValidationError
import subprocess
import tempfile
import shutil
import inspect


# Fallback-safe normalize import
try:
    from nbformat.normalized import normalize
except ImportError:
    def normalize(nb): return nb  # no-op if normalize not available


try:
    import dotenv
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
finally:
    from dotenv import load_dotenv


###########################################################
# Utility Functions
###########################################################

import os
import sys
from pathlib import Path


def detect_jupyter_environment():
    """
    Detects the Jupyter environment and returns one of:
    - "colab": Running in official Google Colab
    - "lightning": Running in Lightning AI Studio
    - "vscode": Running in VSCode
    - "cocalc": Running inside the CoCalc frontend
    - "cocalc_compute_server": Running on a compute server (e.g., GCP or Hyperstack) launched from CoCalc
    - "paperspace": Running in a Paperspace notebook
    - "unknown": Environment not recognized
    """
    import os
    import sys
    from pathlib import Path

    # --- Check for Lightning AI Studio ---
    if (
        os.environ.get('ENVIRONMENT', '').lower().startswith('beta-lightning')
        or 'LIGHTNING_CLOUDSPACE_HOST' in os.environ
        or 'LIGHTNING_RESOURCE_TYPE' in os.environ
    ):
        return "lightning"

    # --- Check for CoCalc frontend ---
    if 'COCALC_CODE_PORT' in os.environ:
        return "cocalc"
    
    # --- Check for CoCalc Compute Server (e.g., GCP or Hyperstack) ---
    if Path.home().joinpath("cs_workspace").exists():
        return "cocalc_compute_server"

    # --- Check for Google Colab ---
    if 'google.colab' in sys.modules:
        if 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_GPU' in os.environ:
            return "colab"

    # --- Check for VSCode ---
    if 'VSCODE_PID' in os.environ:
        return "vscode"

    # --- Check for Paperspace ---
    if 'PAPERSPACE_NOTEBOOK_ID' in os.environ:
        return "paperspace"

    # --- Fallback ---
    return "unknown"


def config_paths_keys(env_path=None, api_env_path=None, local_workspace=False):
    """
    Configures workspace paths and loads API keys based on the runtime environment.

    This is the primary setup function used at the beginning of every lesson and homework
    notebook. It automatically detects the environment (CoCalc, Colab, local), sets up
    appropriate storage paths, and loads API keys for external services.

    Args:
        env_path: Path to environment file (optional)
        api_env_path: Path to API keys file (optional)
        local_workspace: If True, creates workspace in notebook's directory (for testing)

    Returns:
        dict: Dictionary containing configured paths:
            - 'MODELS_PATH': Where trained models are saved
            - 'DATA_PATH': Where datasets are stored
            - 'CACHE_PATH': Where downloaded pretrained models are cached

    Example:
        Basic usage in lesson notebooks (from L01-L05):

        ```python
        from introdl.utils import config_paths_keys

        # Set up paths and load API keys
        paths = config_paths_keys()
        MODELS_PATH = paths['MODELS_PATH']
        DATA_PATH = paths['DATA_PATH']

        # Use in training (L01_2_Nonlinear_Regression_1D)
        checkpoint_file = MODELS_PATH / 'L01_MCurveData_CurveFitter.pt'

        # Use in data loading (L03_1_Optimizers_with_CIFAR10)
        train_loader, test_loader = create_CIFAR10_loaders(data_dir=DATA_PATH)
        ```

        Advanced usage with local workspace:

        ```python
        # For local development/testing
        paths = config_paths_keys(local_workspace=True)
        # Creates home_workspace/ in current directory
        ```

        Environment detection and output example:
        ```
        ✅ Environment: CoCalc Home Server
        📂 Storage Configuration:
           DATA_PATH: ~/home_workspace/data
           MODELS_PATH: ~/Lesson_01_Models (local to this notebook)
           CACHE_PATH: ~/home_workspace/downloads
        🔑 API keys: 2 loaded from home_workspace/api_keys.env
        📦 introdl v1.4.3 ready
        ```
    """
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    # detect_jupyter_environment is defined in this same file - no import needed
    from introdl.paths import (
        get_course_root, get_workspace_dir, resolve_env_file, resolve_api_keys_file
    )
    
    # Check for environment variable override
    if os.environ.get('DS776_LOCAL_WORKSPACE', '').lower() == 'true':
        local_workspace = True

    env_names = {
        "colab": "Official Google Colab",
        "lightning": "Lightning AI Studio",
        "cocalc": "CoCalc Home Server",
        "cocalc_compute_server": "CoCalc Compute Server",
        "vscode": "VSCode Jupyter",
        "paperspace": "Paperspace Notebook",
        "unknown": "Unknown Environment"
    }

    environment = detect_jupyter_environment()
    
    # ========================================================================
    # NEW: Check for Lesson/Homework directory and create local models folder
    # ========================================================================
    cwd = Path.cwd()
    parent_dir = cwd.name  # Get the current directory name
    local_models_dir = None
    
    # Check if we're in a Lesson or Homework folder
    if parent_dir.startswith("Lesson_") or parent_dir.startswith("Homework_"):
        try:
            if parent_dir.startswith("Lesson_"):
                # Extract lesson number from directory name
                # e.g., "Lesson_07_Transformers_Intro" -> "Lesson_07_Models"
                parts = parent_dir.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    lesson_num = parts[1]  # Keep zero-padding
                    local_models_dir = cwd / f"Lesson_{lesson_num}_Models"
                    
            elif parent_dir.startswith("Homework_"):
                # Extract homework number from directory name
                # e.g., "Homework_07" -> "Homework_07_Models"
                parts = parent_dir.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    hw_num = parts[1]  # Keep zero-padding
                    local_models_dir = cwd / f"Homework_{hw_num}_Models"
            
            if local_models_dir:
                # Create the directory if it doesn't exist
                local_models_dir.mkdir(exist_ok=True)
                    
        except Exception as e:
            # Silent fallback - don't confuse students with error messages
            local_models_dir = None
    
    # Handle local workspace mode (for testing student solutions)
    if local_workspace:
        print(f"📦 Local Workspace Mode - Creating workspace in notebook directory")
        # Try to get notebook directory, fallback to current working directory
        try:
            import ipykernel
            import json
            from pathlib import Path
            
            # Get the notebook's directory
            connection_file = ipykernel.get_connection_file()
            with open(connection_file) as f:
                kernel_data = json.load(f)
            
            # This is imperfect but works for most cases
            notebook_dir = Path.cwd()
        except:
            notebook_dir = Path.cwd()
        
        # Create home_workspace in the notebook's directory
        base_path = notebook_dir / "home_workspace"
        data_path = base_path / "data"
        
        # Use local models directory if available, otherwise use home_workspace/models
        if local_models_dir:
            models_path = local_models_dir
        else:
            models_path = base_path / "models"
            
        cache_path = base_path / "downloads"
        
        for path in [data_path, models_path, cache_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Format path for display in local workspace mode  
        def format_path_for_display(path):
            """Format path for display with cleaner output"""
            path_str = str(path)
            home_str = str(Path.home())
            
            # If DS776_ROOT_DIR is set (local development), shorten the path
            if 'DS776_ROOT_DIR' in os.environ:
                root_dir = os.environ['DS776_ROOT_DIR']
                # Replace the root directory portion with <DS776_ROOT_DIR>
                if path_str.startswith(root_dir):
                    path_str = path_str.replace(root_dir, "<DS776_ROOT_DIR>")
            # For all environments, replace home directory with ~ for brevity
            elif path_str.startswith(home_str):
                path_str = path_str.replace(home_str, "~")
            
            return path_str
        
        print(f"   Workspace created at: {format_path_for_display(base_path)}")
        
        # Set environment variables
        os.environ["DATA_PATH"] = str(data_path)
        os.environ["MODELS_PATH"] = str(models_path)
        os.environ["CACHE_PATH"] = str(cache_path)
        
        # Configure caching locations
        os.environ["TORCH_HOME"] = str(cache_path)
        os.environ["HF_HOME"] = str(cache_path / "huggingface")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "huggingface" / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "huggingface" / "transformers")
        os.environ["HF_DATASETS_CACHE"] = str(data_path)
        os.environ["XDG_CACHE_HOME"] = str(cache_path)
        os.environ["TQDM_NOTEBOOK"] = "true"
        
    else:
        # Simplified environment detection output
        env_display = env_names.get(environment, 'Unknown')
        if 'DS776_ROOT_DIR' in os.environ:
            print(f"✅ Environment: {env_display} | Course root: {get_course_root()}")
        else:
            print(f"✅ Environment: {env_display}")

    # -- Path setup --
    if local_workspace:
        # Already handled above
        pass
    elif environment in {"colab", "lightning"}:
        # Treat Lightning like Colab: Create local temp_workspace
        base_path = get_workspace_dir(environment)
        data_path = base_path / "data"
        
        # Use local models directory if available
        if local_models_dir:
            models_path = local_models_dir
        else:
            models_path = base_path / "models"
            
        cache_path = base_path / "downloads"

        for path in [data_path, models_path, cache_path]:
            path.mkdir(parents=True, exist_ok=True)

        os.environ["DATA_PATH"] = str(data_path)
        os.environ["MODELS_PATH"] = str(models_path)
        os.environ["CACHE_PATH"] = str(cache_path)
        # Configure ALL caching locations to use our cache_path
        # This centralizes all downloaded models in one place for easy cleanup
        
        # PyTorch/torchvision pretrained models
        os.environ["TORCH_HOME"] = str(cache_path)
        
        # HuggingFace models and datasets
        os.environ["HF_HOME"] = str(cache_path / "huggingface")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "huggingface" / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "huggingface" / "transformers")
        os.environ["HF_DATASETS_CACHE"] = str(data_path)  # Datasets go to data_path
        
        # General cache location (used by some libraries as fallback)
        os.environ["XDG_CACHE_HOME"] = str(cache_path)
        
        # TQDM settings
        os.environ["TQDM_NOTEBOOK"] = "true"

    else:
        # Determine paths based on environment (CoCalc base vs compute server vs local)
        home_dir = Path.home()
        cs_workspace = home_dir / "cs_workspace"
        home_workspace = home_dir / "home_workspace"
        
        # Check if we're on a CoCalc compute server (has cs_workspace)
        on_compute_server = cs_workspace.exists() and environment == "cocalc_compute_server"
        
        # Check if DS776_ROOT_DIR is set (local development)
        if 'DS776_ROOT_DIR' in os.environ:
            # Local development mode - mirror CoCalc structure
            base_dir = get_course_root()
            home_workspace = base_dir / "home_workspace"
            home_workspace.mkdir(parents=True, exist_ok=True)
            
            data_path = home_workspace / "data"
            cache_path = home_workspace / "downloads"
            
            # Helper function to format paths for display (defined here for early use)
            def format_path_for_display(path):
                """Format path for display with cleaner output"""
                path_str = str(path)
                home_str = str(Path.home())
                
                # If DS776_ROOT_DIR is set (local development), shorten the path
                if 'DS776_ROOT_DIR' in os.environ:
                    root_dir = os.environ['DS776_ROOT_DIR']
                    # Replace the root directory portion with <DS776_ROOT_DIR>
                    if path_str.startswith(root_dir):
                        path_str = path_str.replace(root_dir, "<DS776_ROOT_DIR>")
                # For CoCalc environments, replace home directory with ~
                elif environment in ["cocalc", "cocalc_compute_server"] and path_str.startswith(home_str):
                    path_str = path_str.replace(home_str, "~")
                
                return path_str
            
            # Environment already printed above, just show workspace
            print(f"   Using workspace: {format_path_for_display(home_workspace)}")
            
            # Update environment variables
            os.environ["DATA_PATH"] = str(data_path)
            os.environ["CACHE_PATH"] = str(cache_path)
            
        elif environment in ["cocalc", "cocalc_compute_server"]:
            # CoCalc environments
            if on_compute_server:
                # On compute server: use cs_workspace for both data and cache (local, not synced)
                data_path = cs_workspace / "data"
                cache_path = cs_workspace / "downloads"  # Local to compute server, not synced
            else:
                # Base CoCalc: everything in home_workspace
                data_path = home_workspace / "data"
                cache_path = home_workspace / "downloads"
                
            # Update environment variables
            os.environ["DATA_PATH"] = str(data_path)
            os.environ["CACHE_PATH"] = str(cache_path)
            
        else:
            # Other environments (VSCode, unknown, etc.)
            # Check for environment file
            env_file = resolve_env_file(env_path, environment)
            
            if env_file.exists():
                load_dotenv(env_file, override=False)
                print(f"   Loaded config from: {env_file}")
            
            # Use environment variables or defaults
            data_path = Path(os.getenv("DATA_PATH", "~/data")).expanduser()
            cache_path = Path(os.getenv("CACHE_PATH", "~/downloads")).expanduser()
        
        # Models always go in local Lesson/Homework folder if available
        if local_models_dir:
            models_path = local_models_dir
            os.environ["MODELS_PATH"] = str(models_path)
        else:
            # Fallback to environment variable or default
            if 'DS776_ROOT_DIR' in os.environ:
                models_path = get_course_root() / "home_workspace" / "models"
            else:
                models_path = Path(os.getenv("MODELS_PATH", "~/models")).expanduser()
            os.environ["MODELS_PATH"] = str(models_path)

        for path in [data_path, models_path, cache_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Configure ALL caching locations to use our cache_path
        # This centralizes all downloaded models in one place for easy cleanup
        
        # PyTorch/torchvision pretrained models
        os.environ["TORCH_HOME"] = str(cache_path)
        
        # HuggingFace models and datasets
        os.environ["HF_HOME"] = str(cache_path / "huggingface")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "huggingface" / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "huggingface" / "transformers")
        os.environ["HF_DATASETS_CACHE"] = str(data_path)  # Datasets go to data_path
        
        # General cache location (used by some libraries as fallback)
        os.environ["XDG_CACHE_HOME"] = str(cache_path)

    # More verbose path output for clarity
    # Use the format_path_for_display function if it was defined (in local dev mode)
    # Otherwise, define a simple version for other environments
    if 'DS776_ROOT_DIR' not in os.environ and 'format_path_for_display' not in locals():
        # Define the function here for non-local-dev environments
        def format_path_for_display(path):
            """Format path for display with cleaner output"""
            path_str = str(path)
            home_str = str(Path.home())
            
            # Replace home directory with ~ for brevity in all environments
            if path_str.startswith(home_str):
                path_str = path_str.replace(home_str, "~")
            
            return path_str
    print(f"\n📂 Storage Configuration:")
    
    # Special handling for compute servers - split synced vs local
    if environment == "cocalc_compute_server":
        # Models are synced with CoCalc home server
        print(f"   Synced with CoCalc Home Server:")
        print(f"      MODELS_PATH: {format_path_for_display(models_path)}")
        print(f"      ⚠️ Shared storage limited to about 10GB")
        
        # Data and cache are local to compute server only
        print(f"   Local only on this Compute Server:")
        print(f"      DATA_PATH: {format_path_for_display(data_path)}")
        print(f"      CACHE_PATH: {format_path_for_display(cache_path)}")
        print(f"      ℹ️ Additional compute server storage limited to about 50GB")
    else:
        # Regular display for other environments
        print(f"   DATA_PATH: {format_path_for_display(data_path)}")
        if local_models_dir:
            print(f"   MODELS_PATH: {format_path_for_display(models_path)} (local to this notebook)")
        else:
            print(f"   MODELS_PATH: {format_path_for_display(models_path)}")
        print(f"   CACHE_PATH: {format_path_for_display(cache_path)}")
        
        if environment == "cocalc":
            print(f"   ⚠️ 10GB storage limit in CoCalc")

    # -- Load API keys with priority system --
    # Priority: 1) Environment variables, 2) ~/api_keys.env, 3) home_workspace/api_keys.env
    
    # First, capture which API keys/tokens are already in the environment
    existing_keys = {}
    key_patterns = ["_API_KEY", "_TOKEN"]
    # Common placeholder patterns - check for these strings in the key values
    # NOTE: Don't include empty string here - it matches everything!
    placeholder_patterns = ["abcdefg", "your_", "xxx", "replace_me", "enter_your", "paste_here"]

    def is_placeholder_value(val):
        """Check if a value looks like a placeholder rather than a real API key."""
        if not val or len(val) < 10:  # Real API keys are longer than 10 chars
            return True
        val_lower = str(val).lower()
        # Check if it contains common placeholder patterns
        return any(pattern in val_lower for pattern in placeholder_patterns)

    for key in os.environ:
        for pattern in key_patterns:
            if pattern in key:
                val = os.environ[key]
                # Check if it's a real value (not placeholder)
                if not is_placeholder_value(val):
                    existing_keys[key] = val
    
    # Find the API keys file using path_utils (implements priority)
    api_keys_file = resolve_api_keys_file(api_env_path)
    
    # Special handling for Colab - check Google Drive location
    if environment == "colab" and not api_keys_file:
        colab_path = Path("/content/drive/MyDrive/Colab Notebooks/api_keys.env")
        if colab_path.exists():
            api_keys_file = colab_path
    
    # Debug output if requested via environment variable
    if os.environ.get('DEBUG_API_KEYS', '').lower() == 'true':
        print(f"🔍 DEBUG: API key file search:")
        print(f"   Found file: {api_keys_file if api_keys_file else 'None'}")
        if api_keys_file:
            print(f"   File exists: {api_keys_file.exists()}")
    
    if api_keys_file and api_keys_file.exists():
        # Load API keys but don't override existing environment variables
        # This respects the priority: env vars > file values
        load_dotenv(api_keys_file, override=False)
        
        # Clean up any placeholder values that got loaded
        for key in list(os.environ.keys()):
            for pattern in key_patterns:
                if pattern in key:
                    val = os.environ[key]
                    if is_placeholder_value(val):
                        # Remove placeholder values
                        del os.environ[key]

        # Count valid keys loaded
        valid_keys = 0
        for key in os.environ:
            for pattern in key_patterns:
                if pattern in key:
                    val = os.environ[key]
                    if not is_placeholder_value(val):
                        valid_keys += 1
        
        # Condensed API key loading message
        if valid_keys > 0:
            # Show where keys were loaded from with accurate path
            try:
                # Try to show relative path from home
                rel_path = api_keys_file.relative_to(Path.home())
                location = f"~/{rel_path}"
            except ValueError:
                # If not under home, show the full path
                if "home_workspace" in str(api_keys_file):
                    location = "home_workspace/api_keys.env"
                elif api_keys_file.parent == Path.home():
                    location = "~/api_keys.env"
                else:
                    location = str(api_keys_file)
            print(f"🔑 API keys: {valid_keys} loaded from {location}")

    # -- List loaded API keys (condensed) --
    found_keys = []
    for key in os.environ:
        if key.endswith("_API_KEY") or key.endswith("_TOKEN"):
            val = os.environ[key]
            if not is_placeholder_value(val):
                found_keys.append(key)

    if found_keys:
        # Just show count and names, not values
        key_names = ', '.join(sorted(found_keys))
        if len(key_names) > 50:  # Truncate if too long
            key_names = ', '.join(sorted(found_keys)[:3]) + f"... ({len(found_keys)} total)"
        print(f"🔐 Available: {key_names}")

    # -- Hugging Face login if token available --
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and not is_placeholder_value(hf_token):
        try:
            import logging
            import warnings
            # Suppress all HF hub warnings and info messages
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._token").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub._login").setLevel(logging.ERROR)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Environment variable.*HF_TOKEN.*")
                warnings.filterwarnings("ignore", message=".*Note: Environment variable.*")
                warnings.filterwarnings("ignore", category=UserWarning)
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
            print("✅ HuggingFace Hub: Logged in")
        except Exception as e:
            # Silently fail if HF login doesn't work
            pass

    # -- Check OpenRouter credit if API key available --
    if 'OPENROUTER_API_KEY' in found_keys:
        try:
            from .nlp import update_openrouter_credit
            credit = update_openrouter_credit()
            if credit is not None:
                print(f"💰 OpenRouter credit: ${credit:.2f}")
        except Exception:
            # Silently fail if credit check doesn't work
            pass

    # Print version (condensed)
    try:
        import introdl
        print(f"📦 introdl v{introdl.__version__} ready\n")
    except:
        pass
    
    return {
        'MODELS_PATH': models_path,
        'DATA_PATH': data_path,
        'CACHE_PATH': cache_path
    }


def check_cache_usage():
    """
    Check disk usage of cache directories.
    Returns a dictionary with size information.
    """
    import subprocess
    from pathlib import Path
    
# === REMOVED_FUNCTION_START: get_dir_size ===
#     def get_dir_size(path):
#         """Get directory size in bytes"""
#         if not path.exists():
#             return 0
#         try:
#             result = subprocess.run(['du', '-sb', str(path)], 
#                                   capture_output=True, text=True, timeout=10)
#             if result.returncode == 0:
#                 return int(result.stdout.split()[0])
#         except:
#             pass
#         return 0
# === REMOVED_FUNCTION_END: get_dir_size ===

    
    def format_size(size_bytes):
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    cache_path = Path(os.getenv("CACHE_PATH", "~/downloads")).expanduser()
    models_path = Path(os.getenv("MODELS_PATH", "~/models")).expanduser()
    data_path = Path(os.getenv("DATA_PATH", "~/data")).expanduser()
    
    sizes = {}
    
    # Check cache subdirectories
    if cache_path.exists():
        sizes['cache_total'] = get_dir_size(cache_path)
        sizes['pytorch_cache'] = get_dir_size(cache_path / "hub")
        sizes['huggingface_cache'] = get_dir_size(cache_path / "huggingface")
    else:
        sizes['cache_total'] = 0
        sizes['pytorch_cache'] = 0
        sizes['huggingface_cache'] = 0
    
    # Check other directories
    sizes['student_models'] = get_dir_size(models_path) if models_path.exists() else 0
    sizes['datasets'] = get_dir_size(data_path) if data_path.exists() else 0
    
    # Print summary
    print("📊 Storage Usage Summary:")
    print("-" * 40)
    print(f"Student Models ({models_path.name}): {format_size(sizes['student_models'])}")
    print(f"Datasets ({data_path.name}): {format_size(sizes['datasets'])}")
    print(f"Model Cache ({cache_path.name}): {format_size(sizes['cache_total'])}")
    if sizes['pytorch_cache'] > 0:
        print(f"  - PyTorch models: {format_size(sizes['pytorch_cache'])}")
    if sizes['huggingface_cache'] > 0:
        print(f"  - HuggingFace models: {format_size(sizes['huggingface_cache'])}")
    print("-" * 40)
    print(f"Total: {format_size(sum(sizes.values()))}")
    
    return sizes

# === REMOVED_FUNCTION_START: clear_model_cache ===
# def clear_model_cache(cache_type="all", dry_run=True):
#     """
#     Clear cached pretrained models.
    
#     Args:
#         cache_type: "all", "pytorch", "huggingface", or "datasets"
#         dry_run: If True, only show what would be deleted
#     """
#     import shutil
#     from pathlib import Path
    
#     cache_path = Path(os.getenv("CACHE_PATH", "~/downloads")).expanduser()
#     data_path = Path(os.getenv("DATA_PATH", "~/data")).expanduser()
    
#     paths_to_clear = []
    
#     if cache_type in ["all", "pytorch"]:
#         pytorch_cache = cache_path / "hub"
#         if pytorch_cache.exists():
#             paths_to_clear.append(("PyTorch cache", pytorch_cache))
    
#     if cache_type in ["all", "huggingface"]:
#         hf_cache = cache_path / "huggingface"
#         if hf_cache.exists():
#             paths_to_clear.append(("HuggingFace cache", hf_cache))
    
#     if cache_type in ["all", "datasets"]:
#         if data_path.exists():
#             paths_to_clear.append(("Datasets", data_path))
    
#     if dry_run:
#         print("🔍 Would delete the following:")
#         for name, path in paths_to_clear:
#             if path.exists():
#                 size = check_cache_usage().get(name.lower().replace(" ", "_"), 0)
#                 print(f"  - {name}: {path}")
#     else:
#         print("🗑️ Clearing cache...")
#         for name, path in paths_to_clear:
#             if path.exists():
#                 print(f"  Removing {name}...")
#                 shutil.rmtree(path)
#                 path.mkdir(parents=True, exist_ok=True)
#         print("✅ Cache cleared!")
# === REMOVED_FUNCTION_END: clear_model_cache ===


def get_device():
    """Returns the appropriate device ('cuda', 'mps', or 'cpu') depending on availability.

    Automatically detects the best available compute device for PyTorch operations.
    CUDA for NVIDIA GPUs, MPS for Apple Silicon Macs, or CPU as fallback.
    Used in every training notebook to ensure models run on the fastest available hardware.

    Returns:
        torch.device: The best available device for computation.

    Example:
        Standard usage in all lesson notebooks (L01-L05):

        ```python
        from introdl.utils import get_device

        # Get the best available device
        device = get_device()
        print(f"Using device: {device}")

        # Move model to device (L01_2_Nonlinear_Regression_1D)
        model = CurveFitter()
        model.to(device)

        # Use in training (L02_3_MNIST_CNN)
        results_df = train_network(model, loss_func, train_loader,
                                   device=device, epochs=50)

        # Move data to device for inference (L02_1_MNIST_FC)
        inputs = inputs.to(device)
        outputs = model(inputs)
        ```

        Output examples:
        ```
        Using device: cuda        # CoCalc compute servers with GPU
        Using device: mps         # Apple Silicon Macs
        Using device: cpu         # CPU-only systems or base CoCalc
        ```"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
# === REMOVED_FUNCTION_START: cleanup_torch ===
# def cleanup_torch(*objects):
#     """Delete objects, clear CUDA cache, and run garbage collection."""
#     for obj in objects:
#         try:
#             del obj
#         except:
#             pass
#     torch.cuda.empty_cache()
#     gc.collect()
# === REMOVED_FUNCTION_END: cleanup_torch ===



def hf_download(checkpoint_file, repo_id, token=None):
    """
    Download a file directly from the Hugging Face repository.

    Parameters:
    - checkpoint_file (str): The path to the local file where the downloaded file will be saved.
    - repo_id (str): Hugging Face repository ID.
    - token (str, optional): Hugging Face access token for private repositories.

    Returns:
    - None: The file is saved directly to the checkpoint_file location.
    """
    import os
    import requests

    # Construct the file download URL
    base_url = "https://huggingface.co"
    filename = os.path.basename(checkpoint_file)
    file_url = f"{base_url}/{repo_id}/resolve/main/{filename}"

    # Download the file directly
    response = requests.get(file_url, stream=True, headers={"Authorization": f"Bearer {token}"} if token else {})
    if response.status_code != 200:
        raise FileNotFoundError(f"Failed to download '{filename}' from {file_url}. Status code: {response.status_code}")

    # Write the file to the desired checkpoint_file location
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def load_results(checkpoint_file, device=torch.device('cpu'), repo_id="hobbes99/DS776-models", token=None):
    """
    Load the results from a checkpoint file.

    Loads training results (loss, metrics, etc.) from a checkpoint file created during
    model training. Automatically downloads from HuggingFace if not found locally.
    Essential for analyzing training progress without retraining models.

    Args:
        checkpoint_file (str): The path to the checkpoint file.
        device (torch.device, optional): The device to load the checkpoint onto. Defaults to 'cpu'.
        repo_id (str, optional): Hugging Face repository ID for downloading if not found locally.
        token (str, optional): Hugging Face access token for private repositories.

    Returns:
        pd.DataFrame: Training results with columns like 'epoch', 'train loss', 'test loss', etc.

    Example:
        Analyze training results (L01_2_Nonlinear_Regression_1D):

        ```python
        from introdl.utils import load_results

        # Load results from checkpoint
        results_df = load_results(MODELS_PATH / 'L01_MCurveData_CurveFitter.pt')
        print(results_df.tail())  # Show final epochs

        # Check convergence
        final_train_loss = results_df['train loss'].iloc[-1]
        final_test_loss = results_df['test loss'].iloc[-1]
        print(f"Final train/test loss: {final_train_loss:.3f}/{final_test_loss:.3f}")
        ```

        Use with visualization (L02_3_MNIST_CNN):
        ```python
        # Load and plot training curves
        results_df = load_results(MODELS_PATH / 'L02_MNIST_CNN.pt')
        plot_training_metrics(results_df, [['train loss', 'test loss']])

        # Check for overfitting
        gap = results_df['test loss'].iloc[-1] - results_df['train loss'].iloc[-1]
        if gap > 0.1:
            print("⚠️ Possible overfitting detected")
        ```

        Compare experiments (L03_1_Optimizers_with_CIFAR10):
        ```python
        # Load and compare different optimizers
        sgd_results = load_results(MODELS_PATH / 'cifar10_sgd.pt')
        adam_results = load_results(MODELS_PATH / 'cifar10_adam.pt')

        # Find best performing
        sgd_best = sgd_results['test loss'].min()
        adam_best = adam_results['test loss'].min()
        print(f"Best test loss - SGD: {sgd_best:.3f}, Adam: {adam_best:.3f}")
        ```
    """

    # Download the file if it does not exist locally
    if not os.path.exists(checkpoint_file):
        if repo_id is None:
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_file}' not found locally, and no repo_id provided.")
        hf_download(checkpoint_file, repo_id, token)

    # Suppress FutureWarning during torch.load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)

    # Extract the results
    if 'results' not in checkpoint_dict:
        raise KeyError("Checkpoint does not contain 'results'.")
    return pd.DataFrame(checkpoint_dict['results'])

def load_model(model, checkpoint_file, device=torch.device('cpu'), repo_id="hobbes99/DS776-models", token=None):
    """
    Load the model from a checkpoint file, trying locally first, then downloading if not found.

    Loads a trained PyTorch model from a checkpoint file. Can accept either a model class
    or instance. Automatically downloads from HuggingFace if the checkpoint doesn't exist
    locally. Essential for loading pretrained models and resuming training.

    Args:
        model: The model to load. Can be a class or an instance of nn.Module.
        checkpoint_file (str): The path to the checkpoint file.
        device (torch.device, optional): The device to load the model onto. Defaults to 'cpu'.
        repo_id (str, optional): Hugging Face repository ID for downloading if not found locally.
        token (str, optional): Hugging Face access token for private repositories.

    Returns:
        nn.Module: The loaded model with trained weights on specified device.

    Example:
        Load for inference (L01_2_Nonlinear_Regression_1D):

        ```python
        from introdl.utils import load_model

        # Method 1: Pass model class (most common)
        model = load_model(CurveFitter, MODELS_PATH / 'L01_MCurveData_CurveFitter.pt')

        # Method 2: Pass model instance
        model = CurveFitter()
        model = load_model(model, MODELS_PATH / 'L01_MCurveData_CurveFitter.pt')

        # Use for inference
        model.eval()
        with torch.no_grad():
            x_test = torch.linspace(-6, 6, 201).reshape(-1, 1)
            predictions = model(x_test)
        ```

        Load CNN model (L02_3_MNIST_CNN):
        ```python
        # Load trained CNN for evaluation
        model = load_model(SimpleCNN, MODELS_PATH / 'L02_MNIST_CNN.pt')

        # Use with evaluation functions
        test_predictions = classifier_predict(test_dataset, model, device)
        evaluate_classifier(test_dataset, model, device)
        ```

        Transfer learning (L05_1_Transfer_Learning):
        ```python
        # Load pretrained model for fine-tuning
        pretrained_model = load_model(ResNet18, MODELS_PATH / 'pretrained_resnet18.pt')

        # Continue training from checkpoint
        results_df = train_network(pretrained_model, loss_func, train_loader,
                                  checkpoint_file=new_checkpoint_file)
        ```
    """

    # Download the file if it does not exist locally
    if not os.path.exists(checkpoint_file):
        if repo_id is None:
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_file}' not found locally, and no repo_id provided.")
        hf_download(checkpoint_file, repo_id, token)

    # Instantiate model if a class is passed
    if inspect.isclass(model):
        model = model()
    elif not isinstance(model, nn.Module):
        raise ValueError("The model must be a class or an instance of nn.Module.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)

    if 'model_state_dict' not in checkpoint_dict:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    return model.to(device)


def summarizer(model, input_size, device=torch.device('cpu'), col_width=20, verbose=False, varnames = True, **kwargs):
    """
    Summarizes the given model by displaying the input size, output size, and number of parameters.

    Provides a detailed breakdown of model architecture using torchinfo. Essential for
    debugging tensor shape issues and understanding model complexity. Used in every lesson
    to verify model architecture before training and diagnose shape mismatches.

    Args:
        model: The PyTorch model to summarize.
        input_size (tuple): The input size (batch_size, *data_shape) for the model.
        device (torch.device, optional): Device to run summary on. Defaults to 'cpu'.
        col_width (int, optional): Width of each column in summary table. Defaults to 20.
        verbose (bool, optional): Show full error stack trace if True. Defaults to False.
        varnames (bool, optional): Show variable names in summary. Defaults to True.
        **kwargs: Additional arguments passed to torchinfo.summary.

    Returns:
        None: Prints the model summary table to stdout.

    Example:
        Model architecture verification (L01_2_Nonlinear_Regression_1D):

        ```python
        from introdl.utils import summarizer

        # Create and summarize model
        model = CurveFitter()
        summarizer(model, input_size=(40, 1), col_width=16)

        # Output shows layer-by-layer breakdown:
        # ========================================================================================
        # Layer (type:depth-idx)                   Input Shape      Output Shape     Param #
        # ========================================================================================
        # CurveFitter                              [40, 1]          [40, 1]          --
        # ├─Sequential: 1-1                        [40, 1]          [40, 1]          --
        # │    └─Linear: 2-1                       [40, 1]          [40, 8]          16
        # │    └─Tanh: 2-2                         [40, 8]          [40, 8]          --
        # Total params: 97
        ```

        CNN architecture analysis (L02_3_MNIST_CNN):
        ```python
        # Analyze CNN layer dimensions
        model = SimpleCNN()
        summarizer(model, input_size=(32, 1, 28, 28))

        # Helpful for understanding convolution output sizes
        # Shows how 28x28 input becomes smaller through conv layers
        ```

        Debugging shape errors (L04_1_Advanced_NN):
        ```python
        # Use summarizer to debug shape mismatches
        try:
            summarizer(model, input_size=(64, 3, 32, 32))
        except RuntimeError as e:
            print(f"Shape error detected: {e}")
            # Adjust model or input size accordingly
        ```

        Compare model complexity (L05_1_Transfer_Learning):
        ```python
        # Compare original vs fine-tuned model sizes
        summarizer(pretrained_model, input_size=(32, 3, 224, 224),
                  col_width=25, varnames=True)

        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        ```
    """
    model = model.to(device)
    try:
        colnames = ["input_size", "output_size", "num_params"]
        rowsettings = ["var_names"] if varnames else ["depth"]
        print(summary(model, input_size=input_size, col_width=col_width, row_settings=rowsettings, col_names=colnames, **kwargs))
    except RuntimeError as e:
        if verbose:
            # Print the full stack trace and original error message
            traceback.print_exc()
            print(f"Original Error: {e}")
        else:
            # Display simplified error message with additional message for verbose option
            error_message = str(e).splitlines()[-1].replace("See above stack traces for more details.", "").strip()
            error_message = error_message.replace("Failed to run torchinfo.", "Failed to run all model layers.")
            error_message += " Run again with verbose=True to see stack trace."
            print(f"Error: {error_message}")

def classifier_predict(dataset, model, device, batch_size=32, return_labels=False):
    """
    Collects predictions from a PyTorch dataset using a classification model.

    Efficiently generates predictions for an entire dataset using batch processing.
    Used internally by evaluation functions and for generating prediction lists.
    Essential utility for model assessment in classification tasks.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to evaluate.
        model (torch.nn.Module): Classification model that outputs logits.
        device (torch.device): Device to run evaluation on.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        return_labels (bool, optional): Whether to return ground truth labels. Defaults to False.

    Returns:
        list: Predicted class indices.
        list (optional): Ground truth labels if return_labels=True.

    Note:
        Assumes model outputs logits (not probabilities) and dataset returns (inputs, labels) tuples.

    Example:
        Generate predictions for analysis (L02_3_MNIST_CNN):

        ```python
        from introdl.utils import classifier_predict

        # Load trained model
        model = load_model(SimpleCNN, MODELS_PATH / 'L02_MNIST_CNN.pt')

        # Get predictions for test set
        predictions = classifier_predict(test_dataset, model, device)
        print(f"Predicted classes: {predictions[:10]}")  # First 10 predictions

        # Get predictions with ground truth for comparison
        pred_labels, true_labels = classifier_predict(
            test_dataset, model, device, return_labels=True
        )

        # Calculate accuracy
        accuracy = sum(p == t for p, t in zip(pred_labels, true_labels)) / len(pred_labels)
        print(f"Accuracy: {accuracy:.3f}")
        ```

        Find misclassified examples (L02_1_MNIST_FC):
        ```python
        # Identify misclassified samples
        predictions, labels = classifier_predict(
            test_dataset, model, device,
            return_labels=True, batch_size=64
        )

        # Find indices of misclassified examples
        misclassified_indices = [
            i for i, (pred, true) in enumerate(zip(predictions, labels))
            if pred != true
        ]

        print(f"Misclassified examples: {len(misclassified_indices)}")

        # Visualize some misclassified examples
        if misclassified_indices:
            create_image_grid(test_dataset, nrows=2, ncols=5,
                             indices=misclassified_indices[:10])
        ```

        Batch size optimization (L03_1_Optimizers_with_CIFAR10):
        ```python
        # Use larger batch size for faster evaluation
        predictions = classifier_predict(
            large_test_dataset, model, device,
            batch_size=128  # Larger batch for speed
        )

        # Memory-efficient evaluation for very large datasets
        predictions = classifier_predict(
            huge_dataset, model, device,
            batch_size=16  # Smaller batch to avoid OOM
        )
        ```
    """
    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    
    # Initialize lists to store predictions and ground truth labels
    predictions = []
    ground_truth = [] if return_labels else None

    # Turn off gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move inputs and labels to the specified device
            inputs = inputs.to(device)
            if return_labels:
                labels = labels.to(device)

            # Forward pass through the model
            logits = model(inputs)

            # Get predicted labels (the class with the highest logit)
            preds = torch.argmax(logits, dim=1)

            # Append predictions to the list
            predictions.extend(preds.cpu().tolist())
            # Append ground truth labels if requested
            if return_labels:
                ground_truth.extend(labels.cpu().tolist())

    if return_labels:
        return predictions, ground_truth
    return predictions

def create_CIFAR10_loaders(transform_train=None, transform_test=None, transform_valid=None,
                           valid_prop=0.2, batch_size=64, seed=42, data_dir='./data', 
                           downsample_prop=1.0, num_workers=1, persistent_workers = True, 
                           use_augmentation=False):
    """Create data loaders for the CIFAR10 dataset.

    Creates PyTorch DataLoaders for CIFAR10 with proper train/test splits, normalization,
    and optional data augmentation. Used extensively in L03-L05 for CNN training experiments.
    Handles all the boilerplate for CIFAR10 data preparation.

    Args:
        transform_train (torchvision.transforms.v2.Compose, optional): Training transformations.
        transform_test (torchvision.transforms.v2.Compose, optional): Test transformations.
        transform_valid (torchvision.transforms.v2.Compose, optional): Validation transformations.
        valid_prop (float): Proportion of training set for validation. Defaults to 0.2.
        batch_size (int): Batch size for data loaders. Defaults to 64.
        seed (int): Random seed for reproducibility. Defaults to 42.
        data_dir (str): Directory to download/store CIFAR10 data. Defaults to './data'.
        downsample_prop (float): Proportion of dataset to keep. Defaults to 1.0.
        num_workers (int): Number of worker processes for data loading. Defaults to 1.
        persistent_workers (bool): Keep workers alive between epochs. Defaults to True.
        use_augmentation (bool): Apply data augmentation to training set. Defaults to False.

    Returns:
        tuple: (train_loader, test_loader) or (train_loader, valid_loader, test_loader)
               if validation split is used.

    Example:
        Basic usage (L03_1_Optimizers_with_CIFAR10):

        ```python
        from introdl.utils import create_CIFAR10_loaders

        # Simple train/test split
        train_loader, test_loader = create_CIFAR10_loaders(
            batch_size=64,
            data_dir=DATA_PATH,
            valid_prop=0.0  # No validation split
        )

        print(f"Training batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        ```

        With validation split (L03_4_Schedulers_with_CIFAR10):
        ```python
        # Train/validation/test split
        train_loader, valid_loader, test_loader = create_CIFAR10_loaders(
            batch_size=128,
            valid_prop=0.2,  # 20% for validation
            seed=42
        )

        # Use in training with validation
        results_df = train_network(model, loss_func, train_loader,
                                  val_loader=valid_loader,
                                  test_loader=test_loader)
        ```

        With data augmentation (L03_3_Augmentation_with_CIFAR10):
        ```python
        # Enable data augmentation for better generalization
        train_loader, test_loader = create_CIFAR10_loaders(
            use_augmentation=True,  # Adds random crops, flips, color jitter
            batch_size=64
        )

        # Augmentation is automatically applied during training
        for images, labels in train_loader:
            # Images are randomly augmented each epoch
            break
        ```"""

    # Set default transforms if none are supplied
    mean = (0.4914, 0.4822, 0.4465) 
    std = (0.2023, 0.1994, 0.2010)

    if transform_train is None:
        if use_augmentation:
            transform_train = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Normalize(mean=mean, std=std), 
                transforms.ToPureTensor()   
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=mean, std=std),
                transforms.ToPureTensor()   
            ])
    
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
            transforms.ToPureTensor()   
        ])
        
    # Set validation transform; if None, use transform_test
    if transform_valid is None:
        transform_valid = transform_test

    # Load the full training and test datasets
    train_dataset_full = CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # Generate indices for training and validation if needed
    train_indices, valid_indices = None, None
    if valid_prop and 0 < valid_prop < 1.0:
        total_indices = list(range(len(train_dataset_full)))
        train_indices, valid_indices = train_test_split(
            total_indices,
            test_size=valid_prop,
            random_state=seed,
            shuffle=True
        )

    # Downsample datasets if required
    if downsample_prop < 1.0:
        train_indices = train_indices[:int(downsample_prop * len(train_indices))] if train_indices else None
        valid_indices = valid_indices[:int(downsample_prop * len(valid_indices))] if valid_indices else None

    # Create Subset datasets for training and optionally validation
    train_dataset = Subset(train_dataset_full, train_indices) if train_indices else train_dataset_full
    valid_dataset = Subset(CIFAR10(root=data_dir, train=True, download=True, transform=transform_valid), valid_indices) if valid_indices else None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, persistent_workers=persistent_workers)
    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                  num_workers=num_workers, persistent_workers=persistent_workers)

    if valid_loader:
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, test_loader

# def wrap_print_text(print):
#     """
#     Wraps the given print function to format text with a specified width.
#     This function takes a print function as an argument and returns a new function
#     that formats the text to a specified width before printing. The text is wrapped
#     to 80 characters per line, and long words are broken to fit within the width.
#     Args:
#         print (function): The original print function to be wrapped.
#     Returns:
#         function: A new function that formats text to 80 characters per line and
#                   then prints it using the original print function.
#     Example:
#         wrapped_print = wrap_print_text(print)
#         wrapped_print("This is a very long text that will be wrapped to fit within 80 characters per line.")
#     Adapted from: https://stackoverflow.com/questions/27621655/how-to-overload-print-function-to-expand-its-functionality/27621927"""

#     def wrapped_func(text):
#         if not isinstance(text, str):
#             text = str(text)
#         wrapper = TextWrapper(
#             width=80,
#             break_long_words=True,
#             break_on_hyphens=False,
#             replace_whitespace=False,
#         )
#         return print("\n".join(wrapper.fill(line) for line in text.split("\n")))

#     return wrapped_func

def wrap_print_text(original_print, width=80):
    """
    Wraps the given print function to format text with a specified width.

    Creates a wrapped version of the print function that automatically formats long text
    to fit within specified line width. Useful for displaying long model outputs,
    generated text, and API responses in a readable format. Used in NLP lessons
    for formatting LLM outputs.

    Args:
        original_print (function): The original print function to wrap.
        width (int, optional): Maximum characters per line. Defaults to 80.

    Returns:
        function: Wrapped print function that formats text before printing.

    Example:
        Format long text output (L07_NLP lessons):

        ```python
        from introdl.utils import wrap_print_text

        # Create wrapped print function
        wrapped_print = wrap_print_text(print, width=60)

        # Use for formatting long model outputs
        long_text = "This is a very long piece of text that would normally extend beyond the readable width of a notebook cell and make it difficult to read the content properly."

        # Regular print (hard to read)
        print(long_text)

        # Wrapped print (nicely formatted)
        wrapped_print(long_text)
        # Output:
        # This is a very long piece of text that would normally
        # extend beyond the readable width of a notebook cell and
        # make it difficult to read the content properly.
        ```

        Format LLM responses (L07+ NLP lessons):
        ```python
        # Format generated text responses
        wrapped_print = wrap_print_text(print, width=70)

        response = llm_generate("Explain deep learning", max_length=200)
        wrapped_print(f"LLM Response: {response}")

        # Better formatting for long explanations
        ```

        Custom width for different contexts:
        ```python
        # Narrow format for side-by-side comparisons
        narrow_print = wrap_print_text(print, width=40)

        # Wide format for detailed explanations
        wide_print = wrap_print_text(print, width=100)

        narrow_print("Short format text")
        wide_print("This text will be formatted with a wider line width for more detailed content")
        ```
    """

    def wrapped_func(*args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        wrapper = TextWrapper(
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        wrapped_text = "\n".join(wrapper.fill(line) for line in text.split("\n"))
        return original_print(wrapped_text, **kwargs)

    return wrapped_func

def _guess_notebook_path():
    """
    Guess the current notebook path by looking for the most recently modified .ipynb file in the working dir.
    """
    cwd = Path.cwd()
    candidates = sorted(cwd.glob("*.ipynb"), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise RuntimeError("No .ipynb files found in the current directory.")
    print(f"[INFO] Using notebook: {candidates[0].name}")
    return candidates[0]

def _clean_invalid_outputs(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        new_outputs = []
        for j, output in enumerate(cell.get("outputs", [])):
            fake_nb = {
                "cells": [{
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": 1,
                    "outputs": [output],
                    "source": "",
                    "id": f"cell-{i}-{j}"
                }],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5
            }
            try:
                normalize(fake_nb)
                validate(fake_nb, relax_add_props=True)
                new_outputs.append(output)
            except NotebookValidationError:
                print(f"[WARN] Removed invalid output from cell {i}, output {j}")

        cell["outputs"] = new_outputs

    normalize(nb)

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

def convert_nb_to_html(output_filename="converted.html", notebook_path=None, template="lab"):
    """
    Convert a notebook to HTML using the specified nbconvert template.

    Converts Jupyter notebooks to clean HTML format for submission and sharing.
    Automatically cleans up metadata issues and handles template fallbacks.
    Essential for homework submission process - creates clean HTML versions
    that instructors can easily review.

    Args:
        output_filename (str or Path): Name or path of resulting HTML file. Defaults to "converted.html".
        notebook_path (str or Path, optional): Path to notebook to convert. If None, uses most recent .ipynb in cwd.
        template (str, optional): nbconvert template ("lab" or "classic"). Defaults to "lab".

    Returns:
        None: Creates HTML file at specified output path.

    Notes:
        - Cleans up CoCalc/Colab-specific metadata that breaks nbconvert
        - Filters out invalid outputs missing 'output_type'
        - Automatically falls back to 'classic' template if 'lab' fails
        - Supports Google Drive paths and network locations

    Example:
        Homework submission (All Homework_XX_Utilities):

        ```python
        from introdl.utils import convert_nb_to_html

        # Convert current homework for submission
        convert_nb_to_html("Homework_01_GRADE_THIS_ONE.html")

        # Convert specific notebook
        convert_nb_to_html(
            output_filename="submission.html",
            notebook_path="Homework_01_Assignment.ipynb"
        )

        # Use classic template for better compatibility
        convert_nb_to_html("homework.html", template="classic")
        ```

        Batch conversion (Course utilities):
        ```python
        # Convert multiple notebooks
        notebooks = ["L01_Assignment.ipynb", "L02_Assignment.ipynb"]

        for nb in notebooks:
            output_name = nb.replace(".ipynb", ".html")
            convert_nb_to_html(output_name, nb)
            print(f"Converted {nb} → {output_name}")
        ```

        Handle conversion errors (Troubleshooting):
        ```python
        # nbconvert sometimes fails with widget metadata
        try:
            convert_nb_to_html("homework.html")
        except Exception as e:
            print(f"Conversion failed: {e}")
            # Function automatically tries classic template as fallback
        ```
    """

    # If no notebook path is given, use most recent .ipynb in current directory
    if notebook_path is None:
        candidates = list(Path.cwd().glob("*.ipynb"))
        if not candidates:
            raise FileNotFoundError("No .ipynb files found in current directory.")
        notebook_path = max(candidates, key=lambda f: f.stat().st_mtime)

    output_filename = Path(output_filename)
    if not output_filename.name.endswith(".html"):
        output_filename = output_filename.with_suffix(".html")

    notebook_path = Path(notebook_path).resolve()
    output_dir = output_filename.parent.resolve()
    output_name = output_filename.stem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / notebook_path.name
        shutil.copy2(notebook_path, tmp_path)
        print(f"[INFO] Temporary copy created: {tmp_path}")

        # 🧼 Clean metadata that breaks nbconvert
        try:
            nb = nbformat.read(tmp_path, as_version=4)

            for cell in nb.cells:
                # Remove invalid top-level fields
                cell.pop("id", None)

                # Clean outputs
                if "outputs" in cell:
                    cleaned_outputs = []
                    for output in cell["outputs"]:
                        if not isinstance(output, dict):
                            continue
                        if "output_type" not in output:
                            # Not a valid output object — skip it
                            continue
                        output.pop("errorDetails", None)
                        output.pop("request", None)
                        if "data" in output and "application/vnd.jupyter.widget-view+json" in output["data"]:
                            output["data"].pop("application/vnd.jupyter.widget-view+json", None)
                        cleaned_outputs.append(output)
                    cell["outputs"] = cleaned_outputs

            # Remove broken widget metadata
            if "metadata" in nb and "widgets" in nb["metadata"]:
                del nb["metadata"]["widgets"]

            nbformat.write(nb, tmp_path)
        except Exception as e:
            print(f"[WARNING] Failed to clean notebook metadata: {e}")

        def run_nbconvert(tmpl):
            return subprocess.run(
                [
                    "jupyter", "nbconvert",
                    "--to", "html",
                    "--template", tmpl,
                    "--output", output_name,
                    "--output-dir", str(output_dir),
                    str(tmp_path)
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

        # 🧪 Try nbconvert with the specified template, fallback to classic if it fails
        result = run_nbconvert(template)
        if result.returncode != 0:
            print(f"[WARNING] nbconvert with template '{template}' failed. Retrying with 'classic'...")
            result = run_nbconvert("classic")

        if result.returncode == 0:
            print(f"[SUCCESS] HTML export complete: {output_dir / output_filename.name}")
        else:
            print("[ERROR] nbconvert failed:\n", result.stderr)
