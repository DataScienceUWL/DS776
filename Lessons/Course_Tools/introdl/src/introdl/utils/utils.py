import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
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

# def detect_jupyter_environment():
#     """
#     Detects the Jupyter environment and returns one of:
#     - "colab": Running in official Google Colab
#     - "vscode": Running in VSCode
#     - "cocalc": Running inside the CoCalc frontend
#     - "cocalc_compute_server": Running on a compute server (e.g., GCP or Hyperstack) launched from CoCalc
#     - "paperspace": Running in a Paperspace notebook
#     - "unknown": Environment not recognized
#     """
    
#     # Check for CoCalc frontend (browser UI)
#     if 'COCALC_CODE_PORT' in os.environ:
#         return "cocalc"
    
#     # Check for CoCalc Compute Server (GCP or Hyperstack)
#     # CoCalc compute servers do NOT set COCALC_CODE_PORT, but do provision ~/cs_workspace
#     if Path.home().joinpath("cs_workspace").exists():
#         return "cocalc_compute_server"

#     # Check for official Google Colab
#     if 'google.colab' in sys.modules:
#         if 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_GPU' in os.environ:
#             return "colab"

#     # Check for VSCode
#     if 'VSCODE_PID' in os.environ:
#         return "vscode"

#     # Check for Paperspace
#     if 'PAPERSPACE_NOTEBOOK_ID' in os.environ:
#         return "paperspace"

#     # Fallback
#     return "unknown"

# def config_paths_keys(env_path="~/Lessons/Course_Tools/local.env", api_keys_env="~/Lessons/Course_Tools/api_keys.env"):
#     """
#     Reads environment variables and sets paths.

#     If running in Colab, sets hardcoded /content/temp_workspace paths.
#     Otherwise uses dotenv to load based on environment:
#     - CoCalc: ~/Lessons/Course_Tools/cocalc.env
#     - Local: ~/Lessons/Course_Tools/local.env

#     Also loads API keys from api_keys.env if HF_TOKEN or OPENAI_API_KEY are not already set.

#     Returns:
#         dict: A dictionary with keys 'MODELS_PATH', 'DATA_PATH', and 'CACHE_PATH'.
#     """

#     environment = detect_jupyter_environment()

#     if environment == "colab":
#         # Set Colab-specific paths
#         base_path = Path("/content/temp_workspace")
#         data_path = base_path / "data"
#         models_path = base_path / "models"
#         cache_path = base_path / "downloads"

#         # Set environment variables
#         os.environ['DATA_PATH'] = str(data_path)
#         os.environ['MODELS_PATH'] = str(models_path)
#         os.environ['CACHE_PATH'] = str(cache_path)
#         os.environ['TORCH_HOME'] = str(cache_path)
#         os.environ['HF_HOME'] = str(cache_path)
#         os.environ['HF_DATASETS_CACHE'] = str(data_path)
#         os.environ['TQDM_NOTEBOOK'] = "true"

#         # Create the directories
#         for path in [data_path, models_path, cache_path]:
#             path.mkdir(parents=True, exist_ok=True)

#         print("[INFO] Environment: colab")
#         print(f"DATA_PATH={data_path}")
#         print(f"MODELS_PATH={models_path}")
#         print(f"CACHE_PATH={cache_path}")

#     else:
#         # Load local.env or environment-specific default
#         home_local_env = Path.home() / "local.env"
#         if home_local_env.exists():
#             env_file = home_local_env
#         else:
#             env_file = Path(env_path).expanduser()

#             if not env_file.exists():
#                 # Auto-choose based on environment
#                 if environment == "cocalc_compute_server":
#                     env_file = Path("~/Lessons/Course_Tools/cocalc_compute_server.env").expanduser()
#                 elif environment == "cocalc":
#                     env_file = Path("~/Lessons/Course_Tools/cocalc.env").expanduser()
#                 elif environment == "colab":
#                     env_file = Path("~/Lessons/Course_Tools/google_colab.env").expanduser()
#                 else:
#                     env_file = Path("~/Lessons/Course_Tools/local.env").expanduser()

#         if env_file.exists():
#             load_dotenv(env_file, override=False)
#             print(f"Loaded environment variables from: {env_file}")
#         else:
#             print(f"Warning: environment file not found at {env_file}")

#         # Retrieve and set paths
#         models_path = Path(os.getenv("MODELS_PATH", "")).expanduser()
#         data_path = Path(os.getenv("DATA_PATH", "")).expanduser()
#         cache_path = Path(os.getenv("CACHE_PATH", "")).expanduser()

#         os.environ["TORCH_HOME"] = str(cache_path)
#         os.environ["HF_HOME"] = str(cache_path)
#         os.environ["HF_DATASETS_CACHE"] = str(data_path)

#         for path in [models_path, data_path, cache_path]:
#             if not path.exists():
#                 path.mkdir(parents=True, exist_ok=True)

#         print(f"MODELS_PATH={models_path}")
#         print(f"DATA_PATH={data_path}")
#         print(f"CACHE_PATH={cache_path}")

#     # 🔐 Load API keys (colab-aware)
#     api_keys_file = None
#     home_api_keys_file = Path.home() / "api_keys.env"
#     colab_api_keys_file = Path("/content/drive/MyDrive/Colab Notebooks/api_keys.env")

#     if home_api_keys_file.exists():
#         api_keys_file = home_api_keys_file
#     elif environment == "colab" and colab_api_keys_file.exists():
#         api_keys_file = colab_api_keys_file
#     elif api_keys_env:
#         api_keys_file = Path(api_keys_env).expanduser()

#     if api_keys_file and api_keys_file.exists():
#         load_dotenv(api_keys_file, override=False)
#         print(f"Loaded API keys from: {api_keys_file}")
#     else:
#         print(f"Warning: API keys file not found. Looked in {home_api_keys_file} and {colab_api_keys_file}")

#     # 🔐 Login to Hugging Face
#     if os.getenv("HF_TOKEN"):
#         try:
#             import logging
#             logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
#             from huggingface_hub import login
#             login(token=os.getenv("HF_TOKEN"))
#             print("Successfully logged in to Hugging Face Hub.")
#         except Exception as e:
#             print(f"Failed to login to Hugging Face Hub: {e}")
#     else:
#         print("Set HF_TOKEN in api_keys.env or in the environment to login to Hugging Face Hub")

#     return {
#         'MODELS_PATH': models_path,
#         'DATA_PATH': data_path,
#         'CACHE_PATH': cache_path
#     }

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


# def config_paths_keys(env_path=None, api_env_path=None):
#     """
#     Reads environment variables and sets paths based on runtime environment.

#     Parameters:
#         env_path (str or Path, optional): Path to environment file to load. If None, selects automatically.
#         api_env_path (str or Path, optional): Path to API key env file. If None, searches default locations.

#     Uses the following defaults if env_path is None:
#         - Colab:         ~/Lessons/Course_Tools/google_colab.env
#         - CoCalc:        ~/Lessons/Course_Tools/cocalc.env
#         - CoCalc Server: ~/Lessons/Course_Tools/cocalc_compute_server.env
#         - Default:       ~/Lessons/Course_Tools/local.env

#     Also loads API keys from api_keys.env if not already set.

#     Returns:
#         dict: {'MODELS_PATH', 'DATA_PATH', 'CACHE_PATH'}
#     """
#     from introdl.utils import detect_jupyter_environment  # Assumes available

#     env_names = {"colab": "Official Google Colab",
#                  "cocalc": "CoCalc Home Server",
#                  "cocalc_compute_server": "CoCalc Compute Server",
#                  "vscode": "VSCode Jupyter",
#                  "paperspace": "Paperspace Notebook",
#                  "unknown": "Unknown Environment"}

#     environment = detect_jupyter_environment()
#     print(f"✅ Detected environment: {env_names[environment]}\n")

#     # -- Path configuration --
#     if environment == "colab":
#         base_path = Path("/content/temp_workspace")
#         data_path = base_path / "data"
#         models_path = base_path / "models"
#         cache_path = base_path / "downloads"

#         for path in [data_path, models_path, cache_path]:
#             path.mkdir(parents=True, exist_ok=True)

#         os.environ["DATA_PATH"] = str(data_path)
#         os.environ["MODELS_PATH"] = str(models_path)
#         os.environ["CACHE_PATH"] = str(cache_path)
#         os.environ["TORCH_HOME"] = str(cache_path)
#         os.environ["HF_HOME"] = str(cache_path)
#         os.environ["HF_DATASETS_CACHE"] = str(data_path)
#         os.environ["TQDM_NOTEBOOK"] = "true"

#     else:
#         # Determine which env file to use
#         if env_path is not None:
#             env_file = Path(env_path).expanduser()
#         else:
#             env_map = {
#                 "cocalc_compute_server": "~/Lessons/Course_Tools/cocalc_compute_server.env",
#                 "cocalc": "~/Lessons/Course_Tools/cocalc.env",
#                 "colab": "~/Lessons/Course_Tools/google_colab.env",
#             }
#             env_file = Path(env_map.get(environment, "~/Lessons/Course_Tools/local.env")).expanduser()

#         if env_file.exists():
#             load_dotenv(env_file, override=False)
#             print(f"Loading workspace paths from: {env_file}")
#         else:
#             print(f"⚠️ .env file not found: {env_file}")

#         data_path = Path(os.getenv("DATA_PATH", "~/data")).expanduser()
#         models_path = Path(os.getenv("MODELS_PATH", "~/models")).expanduser()
#         cache_path = Path(os.getenv("CACHE_PATH", "~/cache")).expanduser()

#         for path in [data_path, models_path, cache_path]:
#             path.mkdir(parents=True, exist_ok=True)

#         os.environ["TORCH_HOME"] = str(cache_path)
#         os.environ["HF_HOME"] = str(cache_path)
#         os.environ["HF_DATASETS_CACHE"] = str(data_path)

#     print(f"DATA_PATH={data_path}")
#     print(f"MODELS_PATH={models_path}")
#     print(f"CACHE_PATH={cache_path}")
#     print(f"✅ Workspace paths configured for {env_names[environment]}\n")

#     # -- API Keys loading --
#     api_keys_file = None
#     if api_env_path is not None:
#         api_keys_file = Path(api_env_path).expanduser()
#     else:
#         default_api_paths = [
#             Path.home() / "api_keys.env",
#             Path("/content/drive/MyDrive/Colab Notebooks/api_keys.env") if environment == "colab" else None,
#             Path("~/Lessons/Course_Tools/api_keys.env").expanduser()
#         ]
#         for path in default_api_paths:
#             if path and path.exists():
#                 api_keys_file = path
#                 break

#     if api_keys_file and api_keys_file.exists():
#         load_dotenv(api_keys_file, override=False)
#         print(f"✅ Loaded API keys from: {api_keys_file}\n")
#     else:
#         print("⚠️ No api_keys.env file found in expected locations.\n")


#     # -- List loaded API keys (excluding 'abcdefg') --
#     found_keys = [
#         key for key in os.environ
#         if key.endswith("_API_KEY") and os.environ[key] != "abcdefg"
#     ]
#     if found_keys:
#         print("🔐 API keys loaded (excluding 'abcdefg'):")
#         for key in sorted(found_keys):
#             print(f"  - {key}")
#         print("")
#     else:
#         print("⚠️ No valid *_API_KEY environment variables found (excluding 'abcdefg').")

#     # -- Hugging Face login --
#     hf_token = os.getenv("HF_TOKEN")
#     if hf_token:
#         try:
#             import logging
#             logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
#             from huggingface_hub import login
#             login(token=hf_token)
#             print("✅ Logged into Hugging Face Hub.")
#         except Exception as e:
#             print(f"[ERROR] Hugging Face login failed: {e}")
#     else:
#         print("⚠️ HF_TOKEN not set. Set it in api_keys.env or the environment.")

#     return {
#         'MODELS_PATH': models_path,
#         'DATA_PATH': data_path,
#         'CACHE_PATH': cache_path
#     }

def config_paths_keys(env_path=None, api_env_path=None):
    """
    Configures workspace paths and loads API keys based on the runtime environment.

    Returns:
        dict: {'MODELS_PATH', 'DATA_PATH', 'CACHE_PATH'}
    """
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from introdl.utils import detect_jupyter_environment  # adjust if needed
    from introdl.utils.path_utils import (
        get_course_root, get_workspace_dir, resolve_env_file, resolve_api_keys_file
    )

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
    print(f"✅ Detected environment: {env_names.get(environment, 'Unknown')}")
    
    # Show course root if DS776_ROOT_DIR is set
    if 'DS776_ROOT_DIR' in os.environ:
        print(f"   Using course root: {get_course_root()} (from DS776_ROOT_DIR)")
    print()

    # -- Path setup --
    if environment in {"colab", "lightning"}:
        # Treat Lightning like Colab: Create local temp_workspace
        base_path = get_workspace_dir(environment)
        data_path = base_path / "data"
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
        os.environ["TORCH_HUB"] = str(cache_path / "hub")  # For torch.hub models
        
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
        # Determine base directory for paths
        # If DS776_ROOT_DIR is set, use course root, otherwise use home directory
        if 'DS776_ROOT_DIR' in os.environ:
            base_dir = get_course_root()
            print(f"📁 Using course root for workspace: {base_dir}")
        else:
            base_dir = Path.home()
            print(f"📁 Using home directory for workspace: {base_dir}")
        
        # Check for environment file but don't rely on it for DS776_ROOT_DIR case
        env_file = resolve_env_file(env_path, environment)
        
        # Only load env file if DS776_ROOT_DIR is NOT set
        # (when DS776_ROOT_DIR is set, we want to use course-relative paths)
        if 'DS776_ROOT_DIR' not in os.environ and env_file.exists():
            load_dotenv(env_file, override=False)
            print(f"Loaded workspace paths from: {env_file}")
        
        # Set paths based on whether DS776_ROOT_DIR is available
        if 'DS776_ROOT_DIR' in os.environ:
            # Use course-relative paths for local development
            # Force override any existing env vars when DS776_ROOT_DIR is set
            data_path = base_dir / "data"
            models_path = base_dir / "models"
            cache_path = base_dir / "downloads"
            
            # Update environment variables to match
            os.environ["DATA_PATH"] = str(data_path)
            os.environ["MODELS_PATH"] = str(models_path)
            os.environ["CACHE_PATH"] = str(cache_path)
        else:
            # Use home directory paths for students (or env file values if loaded)
            data_path = Path(os.getenv("DATA_PATH", "~/data")).expanduser()
            models_path = Path(os.getenv("MODELS_PATH", "~/models")).expanduser()
            cache_path = Path(os.getenv("CACHE_PATH", "~/downloads")).expanduser()

        for path in [data_path, models_path, cache_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Configure ALL caching locations to use our cache_path
        # This centralizes all downloaded models in one place for easy cleanup
        
        # PyTorch/torchvision pretrained models
        os.environ["TORCH_HOME"] = str(cache_path)
        os.environ["TORCH_HUB"] = str(cache_path / "hub")  # For torch.hub models
        
        # HuggingFace models and datasets
        os.environ["HF_HOME"] = str(cache_path / "huggingface")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "huggingface" / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "huggingface" / "transformers")
        os.environ["HF_DATASETS_CACHE"] = str(data_path)  # Datasets go to data_path
        
        # General cache location (used by some libraries as fallback)
        os.environ["XDG_CACHE_HOME"] = str(cache_path)

    print(f"DATA_PATH={data_path}")
    print(f"MODELS_PATH={models_path}")
    print(f"CACHE_PATH={cache_path}")
    print(f"✅ Workspace paths configured for {env_names.get(environment, 'Unknown')}\n")

    # -- Load API keys --
    # First, capture which API keys/tokens are already in the environment
    existing_keys = {}
    key_patterns = ["_API_KEY", "_TOKEN"]
    for key in os.environ:
        for pattern in key_patterns:
            if pattern in key:
                existing_keys[key] = os.environ[key]
    
    # Find the API keys file using path_utils
    api_keys_file = resolve_api_keys_file(api_env_path)
    
    # Special handling for Colab - check Google Drive location
    if environment == "colab" and not api_keys_file:
        colab_path = Path("/content/drive/MyDrive/Colab Notebooks/api_keys.env")
        if colab_path.exists():
            api_keys_file = colab_path
    
    if api_keys_file and api_keys_file.exists():
        # Load API keys but don't override existing environment variables
        load_dotenv(api_keys_file, override=False)
        print(f"✅ Loaded API keys from: {api_keys_file}")
        
        # Report which keys were already in environment vs loaded from file
        new_keys = []
        skipped_keys = []
        for key in os.environ:
            for pattern in key_patterns:
                if pattern in key and os.environ[key] != "abcdefg":
                    if key in existing_keys:
                        if existing_keys[key] != "abcdefg":
                            skipped_keys.append(key)
                    else:
                        new_keys.append(key)
        
        if skipped_keys:
            print(f"   Kept existing environment keys: {', '.join(sorted(skipped_keys))}")
        if new_keys:
            print(f"   Loaded new keys from file: {', '.join(sorted(new_keys))}")
        print()
    else:
        print("⚠️ No api_keys.env file found.\n")

    # -- List loaded API keys --
    found_keys = [
        key for key in os.environ
        if (key.endswith("_API_KEY") or key.endswith("_TOKEN")) and os.environ[key] != "abcdefg"
    ]
    
    if found_keys:
        print("🔐 API keys/tokens available (excluding placeholders):")
        for key in sorted(found_keys):
            # Show first 4 chars of the key value for verification (safe for API keys)
            value_preview = os.environ[key][:4] + "..." if len(os.environ[key]) > 4 else "***"
            print(f"  - {key}: {value_preview}")
        print("")
    else:
        print("⚠️ No valid API keys or tokens found (excluding 'abcdefg' placeholders).\n")

    # -- Hugging Face login if token available --
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token != "abcdefg":
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
            print("✅ Logged into Hugging Face Hub.")
        except Exception as e:
            print(f"[ERROR] Hugging Face login failed: {e}")
    else:
        print("⚠️ HF_TOKEN not set or is placeholder. Set it in api_keys.env or the environment.")

    # Print version and cache configuration info
    try:
        import introdl
        print(f"\n📦 introdl package version: {introdl.__version__}")
    except:
        pass
    
    print("\n📂 Model Cache Configuration:")
    print(f"   Student models: {models_path}")
    print(f"   Cached models: {cache_path}")
    print("   - PyTorch models: {}/hub".format(cache_path))
    print("   - HuggingFace models: {}/huggingface".format(cache_path))
    print("   💡 Run Install_and_Clean.ipynb to manage cache space")
    
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
    
    def get_dir_size(path):
        """Get directory size in bytes"""
        if not path.exists():
            return 0
        try:
            result = subprocess.run(['du', '-sb', str(path)], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return int(result.stdout.split()[0])
        except:
            pass
        return 0
    
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

def clear_model_cache(cache_type="all", dry_run=True):
    """
    Clear cached pretrained models.
    
    Args:
        cache_type: "all", "pytorch", "huggingface", or "datasets"
        dry_run: If True, only show what would be deleted
    """
    import shutil
    from pathlib import Path
    
    cache_path = Path(os.getenv("CACHE_PATH", "~/downloads")).expanduser()
    data_path = Path(os.getenv("DATA_PATH", "~/data")).expanduser()
    
    paths_to_clear = []
    
    if cache_type in ["all", "pytorch"]:
        pytorch_cache = cache_path / "hub"
        if pytorch_cache.exists():
            paths_to_clear.append(("PyTorch cache", pytorch_cache))
    
    if cache_type in ["all", "huggingface"]:
        hf_cache = cache_path / "huggingface"
        if hf_cache.exists():
            paths_to_clear.append(("HuggingFace cache", hf_cache))
    
    if cache_type in ["all", "datasets"]:
        if data_path.exists():
            paths_to_clear.append(("Datasets", data_path))
    
    if dry_run:
        print("🔍 Would delete the following:")
        for name, path in paths_to_clear:
            if path.exists():
                size = check_cache_usage().get(name.lower().replace(" ", "_"), 0)
                print(f"  - {name}: {path}")
    else:
        print("🗑️ Clearing cache...")
        for name, path in paths_to_clear:
            if path.exists():
                print(f"  Removing {name}...")
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
        print("✅ Cache cleared!")

def get_device():
    """
    Returns the appropriate device ('cuda', 'mps', or 'cpu') depending on availability.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
def cleanup_torch(*objects):
    """Delete objects, clear CUDA cache, and run garbage collection."""
    for obj in objects:
        try:
            del obj
        except:
            pass
    torch.cuda.empty_cache()
    gc.collect()


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

    Parameters:
    - checkpoint_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the checkpoint onto. Defaults to 'cpu'.
    - repo_id (str, optional): Hugging Face repository ID for downloading the checkpoint if not found locally.
    - token (str, optional): Hugging Face access token for private repositories.

    Returns:
    - results (pd.DataFrame): The loaded results from the checkpoint file.
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

    Parameters:
    - model: The model to load. It can be either a class or an instance of the model.
    - checkpoint_file (str): The path to the checkpoint file.
    - device (torch.device, optional): The device to load the model onto. Defaults to 'cpu'.
    - repo_id (str, optional): Hugging Face repository ID for downloading the checkpoint if not found locally.
    - token (str, optional): Hugging Face access token for private repositories.

    Returns:
    - model: The loaded model from the checkpoint file.
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

    Parameters:
    - model: The model to summarize.
    - input_size (tuple): The input size of the model.
    - device (torch.device, optional): The device to summarize the model on. Defaults to 'cpu'.
    - col_width (int, optional): The width of each column in the summary table. Defaults to 20.
    - verbose (bool, optional): If True, display the full error stack trace; otherwise, show only a simplified error message. Defaults to False.
    - **kwargs: Additional keyword arguments to pass to the summary function.
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
    Optionally returns ground truth labels.

    Assumptions:
        - The model outputs logits for each class (not probabilities or class indices).
        - The dataset returns tuples of (inputs, labels) where labels are integers representing class indices.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset to evaluate.
        model (torch.nn.Module): The classification model. Assumes outputs are logits for each class.
        device (torch.device): The device to run the evaluation on.
        return_labels (bool): Whether to return ground truth labels along with predictions.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        list: Predicted labels (class indices).
        list (optional): Ground truth labels (if return_labels=True).
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
    """
    Create data loaders for the CIFAR10 dataset.

    Args:
        transform_train (torchvision.transforms.v2.Compose, optional): Transformations for the training set. Defaults to standard training transforms if None.
        transform_test (torchvision.transforms.v2.Compose, optional): Transformations for the test set. Defaults to standard test transforms if None.
        transform_valid (torchvision.transforms.v2.Compose, optional): Transformations for the validation set. Defaults to None.
        valid_prop (float or None): Proportion of the training set to use for validation. If 0.0 or None, no validation split is made.
        batch_size (int): Batch size for the data loaders.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory to download/load CIFAR10 data.
        downsample_prop (float): Proportion of the dataset to keep if less than 1. Defaults to 1.0.
        num_workers (int): Number of worker processes to use for data loading.
        use_augmentation (bool): Whether to apply data augmentation to the training set. Defaults to False.

    Returns:
        tuple: Train loader, test loader, and optionally valid loader, along with the datasets.
    """

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
    This function takes a print function as an argument and returns a new function
    that formats the text to a specified width before printing. The text is wrapped
    to the specified number of characters per line, and long words are broken to fit.
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

# def convert_nb_to_html(output_filename="converted.html", notebook_path=None, template="lab"):
#     """
#     Convert a notebook to HTML using the specified nbconvert template.

#     Parameters:
#         output_filename (str or Path): Name or path of the resulting HTML file.
#         notebook_path (str or Path): Path to the notebook to convert. If None, uses most recent .ipynb in cwd.
#         template (str): nbconvert template to use ("lab" or "classic"). Defaults to "lab".

#     Notes:
#         - Cleans up Colab-specific metadata (e.g., 'errorDetails') and broken widget metadata.
#         - Automatically falls back to 'classic' template if 'lab' fails.
#         - Final HTML is written to output_filename (supports Google Drive paths).
#     """

#     # If no notebook path is given, use most recent .ipynb in current directory
#     if notebook_path is None:
#         candidates = list(Path.cwd().glob("*.ipynb"))
#         if not candidates:
#             raise FileNotFoundError("No .ipynb files found in current directory.")
#         notebook_path = max(candidates, key=lambda f: f.stat().st_mtime)

#     output_filename = Path(output_filename)
#     if not output_filename.name.endswith(".html"):
#         output_filename = output_filename.with_suffix(".html")

#     notebook_path = Path(notebook_path).resolve()
#     output_dir = output_filename.parent.resolve()
#     output_name = output_filename.stem

#     with tempfile.TemporaryDirectory() as tmpdir:
#         tmp_path = Path(tmpdir) / notebook_path.name
#         shutil.copy2(notebook_path, tmp_path)
#         print(f"[INFO] Temporary copy created: {tmp_path}")

#         # 🧼 Clean Colab errorDetails and invalid widget metadata
#         try:
#             nb = nbformat.read(tmp_path, as_version=4)
#             for cell in nb.cells:
#                 if "outputs" in cell:
#                     for output in cell["outputs"]:
#                         if isinstance(output, dict):
#                             if "errorDetails" in output:
#                                 del output["errorDetails"]
#                             # Remove broken widget views that trigger KeyError
#                             if "data" in output and "application/vnd.jupyter.widget-view+json" in output["data"]:
#                                 output["data"].pop("application/vnd.jupyter.widget-view+json", None)
#             if "metadata" in nb and "widgets" in nb["metadata"]:
#                 del nb["metadata"]["widgets"]
#             nbformat.write(nb, tmp_path)
#         except Exception as e:
#             print(f"[WARNING] Failed to clean notebook metadata: {e}")

#         def run_nbconvert(tmpl):
#             return subprocess.run(
#                 [
#                     "jupyter", "nbconvert",
#                     "--to", "html",
#                     "--template", tmpl,
#                     "--output", output_name,
#                     "--output-dir", str(output_dir),
#                     str(tmp_path)
#                 ],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True
#             )

#         # 🧪 Try nbconvert with the specified template, fallback to classic if it fails
#         result = run_nbconvert(template)
#         if result.returncode != 0:
#             print(f"[WARNING] nbconvert with template '{template}' failed. Retrying with 'classic'...")
#             result = run_nbconvert("classic")

#         if result.returncode == 0:
#             print(f"[SUCCESS] HTML export complete: {output_dir / output_filename.name}")
#         else:
#             print("[ERROR] nbconvert failed:\n", result.stderr)

def convert_nb_to_html(output_filename="converted.html", notebook_path=None, template="lab"):
    """
    Convert a notebook to HTML using the specified nbconvert template.

    Parameters:
        output_filename (str or Path): Name or path of the resulting HTML file.
        notebook_path (str or Path): Path to the notebook to convert. If None, uses most recent .ipynb in cwd.
        template (str): nbconvert template to use ("lab" or "classic"). Defaults to "lab".

    Notes:
        - Cleans up Colab/CoCalc-specific metadata (e.g., 'id', 'errorDetails', 'request').
        - Filters out invalid outputs missing 'output_type'.
        - Automatically falls back to 'classic' template if 'lab' fails.
        - Final HTML is written to output_filename (supports Google Drive paths).
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
