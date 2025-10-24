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

# Tell transformers not to use TensorFlow (fixes Keras 3 compatibility issues in CoCalc)
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = 'NO'

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

__version__ = "1.6.46"
# Version history:
# 1.6.46 - Enhanced rich display to mimic actual training table format
#          - _format_detailed_metrics_display() now returns pandas DataFrame (not string)
#          - Table shows: Epoch | Training Loss | Validation Loss | LOC | MISC | ORG | PER | Overall metrics
#          - Entity columns display formatted nested dicts matching live training display
#          - Training Loss column included (will be empty for pretend_train=True due to HF limitation)
#          - Maintains backward compatibility with text classification tasks
# 1.6.45 - MAJOR: Implemented rich training display recording and playback for NER models
#          - Added _extract_detailed_metrics() to capture complete nested dictionaries from seqeval
#          - Saves training_history_detailed.json with per-entity metrics (LOC, MISC, ORG, PER)
#          - Added _format_detailed_metrics_display() for rich per-entity metric formatting
#          - Enhanced _simulate_training_with_metrics() to show rich display when available
#          - Automatic detection: uses rich display for nested metrics, simple display for flat metrics
#          - Backward compatible: text classification tasks (Lesson 8) unchanged
#          - Works with local models and HuggingFace Hub models
#          - Students training their own models automatically get rich playback
# 1.6.44 - Enhanced _extract_metrics_dataframe() to drop entirely null columns
#          - Removes confusing empty columns (eval_accuracy, eval_f1, train_loss when not logged)
#          - Only shows metrics that actually have values
#          - Cleaner display for both classification and NER tasks
# 1.6.43 - ACTUALLY fixed TrainerWithPretend._extract_metrics_dataframe() to dynamically capture all metrics
#          - Changed from hardcoded metric keys to dynamic extraction of ALL eval_* keys
#          - Properly displays NER-specific metrics (eval_overall_f1, eval_overall_precision, etc.)
#          - Fixes missing metrics when pretend_train=True for NER tasks
# 1.6.42 - Version increment without actual code fix (changelog only)
# 1.6.41 - Fixed TrainerWithPretend to correctly detect model type from config architecture
#          - Reads config.architectures[0] to determine if TokenClassification or SequenceClassification
#          - Fixes "unexpected keyword argument 'labels'" error for NER models
# 1.6.40 - Fixed TrainerWithPretend to use AutoModel instead of AutoModelForSequenceClassification
#          - Now automatically detects correct model class (TokenClassification, SequenceClassification, etc.)
#          - Fixes batch size mismatch error when loading NER models with pretend_train=True
# 1.6.39 - Re-applied NER metrics fix after file recovery from backup
#          - Same fix as 1.6.38 (eval_overall_f1 check) was lost during Google Drive recovery
# 1.6.38 - Fixed TrainerWithPretend metrics display for NER tasks
#          - _simulate_training_with_metrics now checks for eval_overall_f1 (NER) before eval_accuracy
#          - Handles NER models that have eval_accuracy column but all values are NaN
#          - Displays "Overall F1" for NER tasks, "Accuracy" for classification tasks
# 1.6.37 - Enforced min 1024 tokens for budget-based reasoning (Claude, Gemini)
#          - llm_generate() now automatically enforces max_tokens >= 1024 for budget-based reasoning
#          - Shows informational message when adjusting max_tokens for budget-based models
#          - Note: gpt-oss-120b and gpt-oss-20b remain excluded (only work with reasoning enabled)
# 1.6.36 - CRITICAL FIX: Reasoning mode now uses direct HTTP requests (bypasses OpenAI SDK)
#          - Fixed empty response bug by using requests library instead of OpenAI SDK
#          - OpenAI SDK doesn't support custom top-level params like 'reasoning'
#          - Direct HTTP POST sends reasoning params at top level per OpenRouter API spec
#          - Effort-based: {"reasoning": {"effort": "high"}} for o3, DeepSeek, Llama-4
#          - Budget-based: {"reasoning": {"max_tokens": 2000}} for Claude, Gemini
# 1.6.35 - BROKEN: Attempted to fix reasoning with top-level params via OpenAI SDK (doesn't work)
# 1.6.34 - Added reasoning support for Gemini models (gemini-flash-lite, gemini-flash)
#          - Per OpenRouter docs, Gemini thinking models support max_tokens for reasoning budget
#          - Both use budget-based reasoning (same as Claude)
#          - Models with reasoning support: claude-haiku-4.5, gemini-flash-lite, gemini-flash, deepseek-v3.1, llama-4-maverick
# 1.6.33 - Removed gpt-oss-20b and gpt-oss-120b from curated model list
#          - Testing revealed both models return empty responses for complex reasoning tasks
#          - Models unreliable and unsuitable for student use (generating tokens but returning empty strings)
#          - Curated list now contains 13 reliable models
# 1.6.32 - Fixed reasoning support configuration for gpt-oss models based on testing
#          - Updated gpt-oss-20b and gpt-oss-120b to supports_reasoning: false (no observed effect)
#          - Models with verified reasoning support: claude-haiku-4.5, deepseek-v3.1, llama-4-maverick
#          - Test results documented in Developer/Notes/Reasoning_Mode_Test_Results.md
# 1.6.31 - Added reasoning configuration to OpenRouter model metadata
#          - Updated claude-haiku to anthropic/claude-haiku-4.5 with extended thinking mode
#          - Added reasoning_support field to all models in openrouter_models.json
#          - Models now specify: supports_reasoning, reasoning_type (effort/budget), default_reasoning_enabled
#          - llm_generate() automatically disables reasoning for all models by default (cost optimization)
#          - Explicit enable_reasoning=True required to activate reasoning modes
#          - Initial models with reasoning support: claude-haiku-4.5, deepseek-v3.1, llama-4-maverick, gpt-oss-120b, gpt-oss-20b
# 1.6.30 - Added reasoning/thinking parameter support to llm_generate()
#          - New parameters: enable_reasoning (bool), reasoning_effort (str), reasoning_budget (int)
#          - Provider-aware implementation: effort levels for OpenAI/DeepSeek, token budgets for Anthropic/Gemini
#          - Enables extended thinking for Claude, Gemini, and O3 models on OpenRouter
#          - Automatic provider detection from model_id, graceful degradation for unsupported providers
#          - Maintains backward compatibility: reasoning disabled by default
# 1.6.29 - Added GPU warning for Lessons 07-12 (except 09) and Homeworks 07-12 (except 09)
#          - config_paths_keys() detects Lesson/Homework folders 07-12 (excluding 09) and checks for CUDA
#          - Warns students to use Compute Server for HuggingFace pipeline commands if no GPU detected
#          - Clarifies that llm_generate does not need Compute Server (API calls)
#          - Helps students avoid slow inference and potential issues on CPU-only CoCalc home server
# 1.6.28 - Fixed "I/O operation on closed file" logging errors in Jupyter/CoCalc
#          - Added fix_logging_handlers() utility function to clean up closed logging handlers
#          - config_paths_keys() now automatically fixes logging handlers on startup
#          - Prevents logging errors when using transformers library in Jupyter notebooks
#          - Particularly fixes issues in CoCalc with transformers v4.49.0+
# 1.6.27 - BREAKING CHANGE: Removed hf_repo_id parameter from TrainerWithPretend for cleaner API
#          - Model name now auto-derived from output_dir (e.g., 'Lesson_08_Models/L08_fine_tune_distilbert' → 'L08_fine_tune_distilbert')
#          - Automatically checks hobbes99/DS776-models/{model_name}/ on HF Hub without requiring parameter
#          - Priority unchanged: HF Hub → Local → Train from scratch
#          - Signature now identical to standard HuggingFace Trainer (except pretend_train parameter)
# 1.6.26 - TrainerWithPretend now checks HuggingFace Hub FIRST (before local models) when pretend_train=True
#          - Added hf_repo_id parameter to load instructor-hosted models from HF Hub
#          - Priority: HF Hub → Local → Train from scratch
#          - Automatically caches HF Hub models locally for faster future loads
# 1.6.25 - Export Trainer (TrainerWithPretend) from nlp.py for HuggingFace fine-tuning with pretend_train
#          - Enables smart caching: `from introdl import Trainer` for drop-in replacement of HF Trainer
# 1.6.24 - Added TRANSFORMERS_NO_TF and USE_TF environment variables to fix Keras 3 compatibility in CoCalc
#          - Prevents transformers from trying to use TensorFlow when Keras 3 is installed
# 1.6.23 - Added generic homework storage cleanup utilities
#          - New get_homework_storage_report() auto-detects current homework and analyzes storage
#          - New clear_homework_storage() removes workspace subfolders and old model folders
#          - Special handling for YOLO *.pt files in Homework_06 (cleaned from HW 7+)
#          - Created generic Storage_Cleanup_After_HW.ipynb that works from any homework folder
# 1.6.22 - Enhanced export_this_to_html() with optional notebook_path parameter
#          - Can now specify exact notebook to export: export_this_to_html('path/to/notebook.ipynb')
#          - Skips auto-detection when path is provided
#          - Added helpful message after export suggesting path parameter if wrong file was exported
# 1.6.21 - IMPORTANT: config_paths_keys() now automatically calls init_cost_tracking()
#          - Cost tracking is now automatically initialized when OPENROUTER_API_KEY is found
#          - Students no longer need to manually call init_cost_tracking() in notebooks
#          - Session spending tracking works automatically in all notebooks with OpenRouter API key
# 1.6.20 - Simplified show_session_spending() output
#          - Now shows "Total Spent this session" instead of "Total Cost"
#          - Fetches live credit from OpenRouter API
#          - Shows "Approximate Credit remaining" with disclaimer about potential delay
#          - Removed "Overall Credit Status" section with all-time spending
# 1.6.19 - BREAKING CHANGE: llm_list_models() now returns a dictionary instead of list of tuples
#          - Dictionary keyed by model short name with full metadata (model_id, size, costs, json_schema, etc.)
#          - Enables easy lookups: models['gemini-flash-lite']['cost_in_per_m']
#          - Added examples in L07_1_Getting_Started.ipynb demonstrating dictionary usage
#          - Verbose table display still works as before
# 1.6.18 - Added estimate_cost parameter alias for print_cost in llm_generate()
#          - Added show_session_spending() function to display spending since init_cost_tracking() was called
#          - Session spending tracking now works with _SESSION_START_TIME global variable
# 1.6.17 - Added export_this_to_html() function for automatic notebook detection and HTML export
#          - Flattened all imports in Lesson 7 notebooks to use `from introdl import ...` pattern
#          - Updated CLAUDE.md with import conventions for course notebooks
# 1.6.16 - MAJOR REFACTOR: Simplified multi-provider LLM support in llm_generate()
#          - Removed llm_provider parameter - now uses api_key and base_url instead
#          - Added cost_per_m_in and cost_per_m_out for manual cost tracking (non-OpenRouter providers)
#          - Defaults to OpenRouter with automatic cost tracking (unchanged behavior)
#          - Works with ANY OpenAI-compatible API by passing api_key + base_url
#          - Removed complex _init_client() function, replaced with simple _init_openrouter_client()
#          - Updated L07_Other_APIs.ipynb with simplified multi-provider examples
#          - See Developer/Notes/Multi_Provider_LLM_Guide.md for full documentation
# 1.6.15 - Added deepseek-v3.1 (deepseek/deepseek-chat-v3.1) with automatic reasoning disabled
#          - llm_generate() now automatically disables reasoning for deepseek models (~75% token reduction)
#          - Updated model list: removed free-tier models, added paid variants of llama-3.2-1b and gpt-oss-20b
# 1.6.14 - Fixed show_pricing_table() to handle new model metadata dict format
#          - Extract 'id' field from model_data dict instead of treating it as a string
# 1.6.13 - Added init_cost_tracking() function to properly initialize LLM cost tracking
#          - Call after config_paths_keys() to pre-load pricing and establish credit baseline
#          - Ensures print_cost parameter works correctly in llm_generate()
# 1.6.12 - CRITICAL FIX: Include openrouter_models.json in package installation
#          - Added package-data configuration to pyproject.toml
#          - Fixes "file not found" error when using llm_generate() and related functions
# 1.6.11 - Restored flexible torch/torchvision dependencies (no version constraints on torch)
#          - Accepts pre-installed NVIDIA-optimized builds in Hyperstack
#          - Avoids dependency conflicts during installation
# 1.6.10 - Updated llm_get_credits() to always fetch live credit from OpenRouter API
#          - Now returns actual remaining credit, not just tracker-based estimate
#          - Automatically updates cost tracker with live credit on each call
# 1.6.9 - Added llm_get_credits() function for easy credit balance checking
#         - Returns dict with 'limit', 'usage', and 'remaining' keys
# 1.6.8 - Added automatic Course_Tools directory cleanup in auto_update_introdl.py
#         - Removes CoCalc backup files (*~) and obsolete scripts automatically
#         - Enhanced cleanup_introdl.sh with dynamic paths and Python version detection
#         - Cleanup runs max once per 7 days via timestamp file
# 1.6.7 - Added JSON capability metadata to openrouter_models.json
#         - Created test_model_json_capabilities.py script for automated testing
#         - Enhanced llm_generate() to intelligently use JSON modes based on model capabilities
#         - Added get_model_metadata() function and capability filtering to llm_list_models()
#         - Models now include json_support metadata (json_object, json_schema, strict_schema)
# 1.6.6 - Fixed module organization: removed duplicated viz functions from nlp.py
#         - Text generation viz functions now correctly imported from generation.py
#         - Removed nlp_orig.py (all useful code migrated to nlp.py or generation.py)
# 1.6.5 - (REVERTED) Mistakenly duplicated viz functions to nlp.py
# 1.6.4 - Moved clear_pipeline() and print_pipeline_info() from nlp_orig.py to nlp.py
# 1.6.4 - Moved clear_pipeline() and print_pipeline_info() from nlp_orig.py to nlp.py
#         - Preparing to deprecate nlp_orig.py (only visualization functions remain there)
# 1.6.3 - Added OpenRouter credit fetching: displays actual credit in config_paths_keys(), updates cost tracker
#         - New functions: get_openrouter_credit(), update_openrouter_credit()
#         - Cost tracking now uses actual account credit instead of fixed baseline
# 1.6.2 - Added Out/1M column to show_pricing_table for better cost comparison
# 1.6.1 - CRITICAL FIX: Fixed API key loading bug where empty string in placeholder_values matched all keys
#         - Refactored is_placeholder_value() to properly detect placeholders vs real API keys
#         - API keys now load correctly from api_keys.env file
# 1.6.0 - MAJOR: Refactored nlp.py for OpenRouter-first approach with persistent cost tracking
#         - New llm_generate(model_name, prompts, mode='text'|'json', ...) signature
#         - Session-cached pricing lookups, persistent cost tracking in ~/home_workspace/
#         - JSON mode with schema validation and fallback strategies
#         - Cost warnings at 50%, 75%, 90% of student credit
#         - Support for any OpenRouter model, not just predefined ones
#         - Moved old nlp.py to nlp_orig.py for backward compatibility
# 1.5.19 - Aggressive subdirectory removal, delete __init__.py files, rename dirs if can't delete
# 1.5.18 - Force reinstall if old nested structure detected, check actual pip location first
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
        detect_jupyter_environment,
        print_model_freeze_summary,
        fix_logging_handlers
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
        llm_generate,
        llm_list_models,
        llm_get_credits,
        llm_configure,
        init_cost_tracking,
        display_markdown,
        show_cost_summary,
        show_session_spending,
        show_pricing_table,
        reset_cost_tracker,
        resolve_model_name,
        get_model_metadata,
        get_model_price,
        get_openrouter_credit,
        update_openrouter_credit,
        clear_pipeline,
        print_pipeline_info,
        Trainer
    )

    # Text generation visualization functions
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
        export_this_to_html,
        zip_homework_models,
        get_homework_storage_report,
        clear_homework_storage
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
    "detect_jupyter_environment", "print_model_freeze_summary", "fix_logging_handlers",

    # Training
    "train_network", "train_simple_network", "visualize2DSoftmax",

    # Visualization
    "create_image_grid", "plot_training_metrics", "vis_feature_maps", "vis_feature_maps_widget",
    "interactive_mnist_prediction", "plot_transformed_images", "evaluate_classifier",

    # NLP
    "llm_generate", "llm_list_models", "llm_get_credits", "llm_configure", "init_cost_tracking",
    "display_markdown", "show_cost_summary", "show_session_spending", "show_pricing_table",
    "reset_cost_tracker", "resolve_model_name", "get_model_metadata", "get_model_price",
    "get_openrouter_credit", "update_openrouter_credit",
    "clear_pipeline", "print_pipeline_info", "Trainer",

    # Generation
    "model_report", "generate_top_k_table", "generate_greedy_decoding_table",
    "generate_detailed_beam_search", "generate_top_k_sampling", "generate_top_p_sampling",
    "plot_top_k_distribution", "visualize_conversation",

    # Summarization
    "compute_all_metrics", "print_metrics",

    # Storage (commonly used)
    "display_storage_report", "cleanup_old_cache", "delete_current_lesson_models",
    "export_homework_html_interactive", "export_this_to_html", "zip_homework_models",
    "get_homework_storage_report", "clear_homework_storage",

    # Notebook state management
    "save_state", "load_state", "enable_cell_autosave", "list_states", "delete_states",
    "register_dataloader", "rebuild_registered_dataloaders"
]
