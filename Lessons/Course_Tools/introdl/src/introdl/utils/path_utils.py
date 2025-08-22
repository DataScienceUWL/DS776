"""Path utilities for flexible course directory resolution."""
import os
from pathlib import Path


def get_course_root():
    """
    Get the course root directory, checking DS776_ROOT_DIR environment variable first.
    
    Returns:
        Path: The course root directory
        
    Examples:
        - If DS776_ROOT_DIR is set: uses that path
        - Otherwise: uses home directory (~)
    """
    root_dir = os.environ.get('DS776_ROOT_DIR')
    if root_dir:
        return Path(root_dir).expanduser().resolve()
    return Path.home()


def get_lessons_dir():
    """Get the Lessons directory path."""
    return get_course_root() / "Lessons"


def get_homework_dir():
    """Get the Homework directory path."""
    return get_course_root() / "Homework"


def get_solutions_dir():
    """Get the Solutions directory path."""
    return get_course_root() / "Solutions"


def get_course_tools_dir():
    """Get the Course_Tools directory path."""
    return get_lessons_dir() / "Course_Tools"


def get_workspace_dir(environment=None):
    """
    Get the appropriate workspace directory based on environment.
    
    Args:
        environment: Optional environment override. If None, auto-detects.
        
    Returns:
        Path: The workspace directory path
    """
    if environment is None:
        from introdl.utils import detect_jupyter_environment
        environment = detect_jupyter_environment()
    
    if environment in {"colab", "lightning"}:
        return Path.home() / "temp_workspace"
    elif environment == "cocalc_compute_server":
        return Path.home() / "cs_workspace"
    else:
        return Path.home() / "home_workspace"


def resolve_env_file(env_path=None, environment=None):
    """
    Resolve the environment file path.
    
    Args:
        env_path: Optional explicit path to env file
        environment: Optional environment override
        
    Returns:
        Path: The resolved environment file path
    """
    if env_path:
        return Path(env_path).expanduser().resolve()
    
    if environment is None:
        from introdl.utils import detect_jupyter_environment
        environment = detect_jupyter_environment()
    
    # Check for local.env in home directory first
    home_env = Path.home() / "local.env"
    if home_env.exists():
        return home_env
    
    # Map environments to their default env files
    env_map = {
        "cocalc_compute_server": "cocalc_compute_server.env",
        "cocalc": "cocalc.env",
        "vscode": "local.env",
        "paperspace": "paperspace.env",
        "unknown": "local.env"
    }
    
    env_filename = env_map.get(environment, "local.env")
    return get_course_tools_dir() / env_filename


def resolve_api_keys_file(api_env_path=None):
    """
    Resolve the API keys environment file path.
    
    Priority order:
    1. Explicit api_env_path if provided
    2. DS776_ROOT_DIR/home_workspace/api_keys.env (for local development)
    3. ~/api_keys.env (user's home directory)
    4. ~/home_workspace/api_keys.env (CoCalc student editable, synced)
    
    Args:
        api_env_path: Optional explicit path to API keys file
        
    Returns:
        Path or None: The resolved API keys file path if it exists
    """
    import os
    
    if api_env_path:
        path = Path(api_env_path).expanduser().resolve()
        return path if path.exists() else None
    
    # Check standard locations in priority order
    locations = []
    
    # First check DS776_ROOT_DIR/home_workspace if in local development
    if 'DS776_ROOT_DIR' in os.environ:
        locations.append(get_course_root() / "home_workspace" / "api_keys.env")
    
    # Then check home directory
    locations.append(Path.home() / "api_keys.env")
    
    # Finally check ~/home_workspace (for CoCalc)
    locations.append(Path.home() / "home_workspace" / "api_keys.env")
    
    for location in locations:
        if location.exists():
            return location
    
    return None