from .utils import detect_jupyter_environment
from .utils import get_device
from .utils import load_results
from .utils import load_model
from .utils import summarizer
from .utils import create_CIFAR10_loaders
from .utils import classifier_predict
from .utils import wrap_print_text
from .utils import config_paths_keys
from .utils import cleanup_torch
from .utils import convert_nb_to_html

# Import path utilities
from .path_utils import (
    get_course_root,
    get_lessons_dir,
    get_homework_dir,
    get_solutions_dir,
    get_course_tools_dir,
    get_workspace_dir,
    resolve_env_file,
    resolve_api_keys_file
)

__all__ = [
    "detect_jupyter_environment",
    "get_device",
    "load_results",
    "load_model",
    "summarizer",
    "create_CIFAR10_loaders",
    "classifier_predict",
    "wrap_print_text",
    "config_paths_keys",
    "cleanup_torch",  
    "convert_nb_to_html",
    # Path utilities
    "get_course_root",
    "get_lessons_dir", 
    "get_homework_dir",
    "get_solutions_dir",
    "get_course_tools_dir",
    "get_workspace_dir",
    "resolve_env_file",
    "resolve_api_keys_file"
]
