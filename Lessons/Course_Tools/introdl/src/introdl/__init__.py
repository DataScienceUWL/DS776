"""
DS776 Deep Learning Course Package
Simplified flat structure for easier maintenance and documentation.
"""

# Suppress warnings before any imports
import os
import sys
import warnings
import logging

# Suppress all TensorFlow and CUDA environment messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')  # Keep CUDA but suppress warnings

# Suppress Python warnings
warnings.filterwarnings('ignore', message='.*MessageFactory.*')
warnings.filterwarnings('ignore', message='.*GetPrototype.*')
warnings.filterwarnings('ignore', message='.*cuFFT.*')
warnings.filterwarnings('ignore', message='.*cuDNN.*')
warnings.filterwarnings('ignore', message='.*cuBLAS.*')
warnings.filterwarnings('ignore', message='.*All log messages before absl.*')

# Suppress protobuf warnings (common with TensorFlow/PyTorch conflicts)
warnings.filterwarnings('ignore', module='google.protobuf')
warnings.filterwarnings('ignore', module='tensorflow')

# Suppress logging warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Suppress absl logging before it's even imported
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

# Try to suppress absl if already imported
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    # Also redirect absl warnings to null
    absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')
except (ImportError, AttributeError):
    pass

# Create a context manager to suppress stderr during imports
from contextlib import contextmanager
import io

@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output."""
    old_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stderr = old_stderr

__version__ = "1.5.17"
# Version history:
# 1.5.17 - Complete removal of old nested module structure in auto_update for Hyperstack
# 1.5.16 - Remove import tests from auto_update that timeout due to heavy modules
# 1.5.15 - More aggressive stderr suppression during imports to eliminate all warnings
# 1.5.14 - Fixed incorrect API keys template warning in auto_update script
# 1.5.13 - Improved package detection by directly checking site-packages directories
# 1.5.12 - Fixed timeout issues in auto_update by using pip show instead of importing module
# 1.5.11 - Optimized auto_update script for faster execution when no update needed
# 1.5.10 - Fixed warning filter for AttributeError (not a Warning subclass)
# 1.5.9 - Added comprehensive warning suppression for TensorFlow/CUDA/protobuf messages
# 1.5.8 - Fixed import-time error in summarization.py by lazy-loading evaluate metrics
# 1.5.7 - Improved parameter interaction clarity and best epoch tracking when resuming
# 1.5.6 - Added message reporting which epoch had the best model at end of training
# 1.5.5 - Fixed progress bar to show total epochs correctly when resuming (e.g., 3/15 instead of 1/13)
# 1.5.4 - Simplified train_network with save_last, resume_last, and total_epochs for easy recovery
# 1.5.3 - Added auto-save capabilities to train_network (reverted in 1.5.4 for simpler approach)
# 1.5.2 - Added DataLoader resumption support with register_dataloader and rebuild_registered_dataloaders
# 1.5.1 - Updated storage.py to include notebook_states in storage reports
# 1.5.0 - Added notebook_states module for saving/restoring session state
# 1.4.5 - Fixed relative imports in storage.py after module flattening
# 1.4.4 - Renamed _storage and _paths to storage and paths (no longer private modules)
# 1.4.3 - Flattened package structure, removed unused functions via dependency analysis
# 1.4.2 - Fixed NumPy dependency to <2.0 for compatibility with matplotlib/seaborn
# 1.4.1 - Fixed API key priority (DS776_ROOT_DIR first), cs_workspace for data on compute servers

# Temporarily redirect stderr to suppress all import warnings
import os as _os
_original_stderr = sys.stderr
_devnull = open(_os.devnull, 'w')
sys.stderr = _devnull

try:
    # Core utilities (most commonly used)
    from .utils import (
        config_paths_keys,
        get_device,
        load_model,
        load_results,
        summarizer,
        create_CIFAR10_loaders,
        convert_nb_to_html,
        wrap_print_text,
        classifier_predict,
        detect_jupyter_environment
    )

    # Training functions
    from .idlmam import (
        train_network,
        train_simple_network,
        visualize2DSoftmax
    )

    # Visualization functions
    from .visul import (
        create_image_grid,
        plot_training_metrics,
        vis_feature_maps,
        vis_feature_maps_widget,
        interactive_mnist_prediction,
        plot_transformed_images,
        evaluate_classifier,
        image_to_PIL
    )

    # NLP functions
    from .nlp import (
        llm_configure,
        llm_generate,
        llm_list_models,
        clear_pipeline,
        print_pipeline_info,
        display_markdown
    )

    # Text generation functions
    from .generation import (
        model_report,
        generate_top_k_table,
        generate_greedy_decoding_table,
        generate_detailed_beam_search,
        generate_top_k_sampling,
        generate_top_p_sampling,
        plot_top_k_distribution,
        visualize_conversation
    )

    # Summarization functions
    from .summarization import (
        compute_all_metrics,
        print_metrics
    )

    # Storage utilities
    from .storage import (
        display_storage_report,
        cleanup_old_cache,
        delete_current_lesson_models,
        export_homework_html_interactive,
        zip_homework_models
    )

    # Path utilities
    from .paths import (
        get_course_root,
        get_lessons_dir,
        get_course_tools_dir,
        get_workspace_dir,
        resolve_env_file,
        resolve_api_keys_file
    )

    # Notebook state management
    from .notebook_states import (
        save_state,
        load_state,
        enable_cell_autosave,
        list_states,
        delete_states,
        register_dataloader,
        rebuild_registered_dataloaders
    )
finally:
    # Restore stderr after imports
    sys.stderr = _original_stderr
    _devnull.close()

# Define public API
__all__ = [
    # Core utilities
    "config_paths_keys", "get_device", "load_model", "load_results", "summarizer",
    "create_CIFAR10_loaders", "convert_nb_to_html", "wrap_print_text", "classifier_predict",

    # Training
    "train_network", "train_simple_network", "visualize2DSoftmax",

    # Visualization
    "create_image_grid", "plot_training_metrics", "vis_feature_maps", "vis_feature_maps_widget",
    "interactive_mnist_prediction", "plot_transformed_images", "evaluate_classifier",

    # NLP
    "llm_configure", "llm_generate", "llm_list_models", "clear_pipeline",
    "print_pipeline_info", "display_markdown",

    # Generation
    "model_report", "generate_top_k_table", "generate_greedy_decoding_table",
    "generate_detailed_beam_search", "generate_top_k_sampling", "generate_top_p_sampling",
    "plot_top_k_distribution", "visualize_conversation",

    # Summarization
    "compute_all_metrics", "print_metrics",

    # Storage (commonly used)
    "display_storage_report", "cleanup_old_cache", "delete_current_lesson_models",
    "export_homework_html_interactive", "zip_homework_models",

    # Notebook state management
    "save_state", "load_state", "enable_cell_autosave", "list_states", "delete_states",
    "register_dataloader", "rebuild_registered_dataloaders"
]
