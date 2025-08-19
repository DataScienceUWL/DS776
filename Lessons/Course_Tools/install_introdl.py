import sys
import subprocess
from pathlib import Path

def find_course_tools_path():
    """
    Finds the 'Course_Tools' directory by searching upwards from the current directory.
    This makes the path resolution independent of the project's root location.
    """
    current_path = Path.cwd()
    while current_path != current_path.parent: # Stop at the filesystem root
        target_path = current_path / 'Course_Tools'
        if target_path.is_dir():
            return target_path.resolve()
        current_path = current_path.parent
    return None # Return None if not found

# A global flag to ensure this function only runs once per session
__COURSE_PACKAGE_CHECKED = False

def install_course_package():
    """
    Checks if the 'introdl' package is installed. If not, installs it
    and instructs the user to restart the kernel. Runs only once per session.
    """
    global __COURSE_PACKAGE_CHECKED
    if __COURSE_PACKAGE_CHECKED:
        return

    try:
        import introdl
        print("✅ The 'introdl' package is already installed and ready to use.")
        
    except ImportError:
        print("🛠️ The 'introdl' package was not found. Attempting to install now...")
        
        course_tools_dir = find_course_tools_path()
        if course_tools_dir is None:
            print("❌ Error: Could not find the 'Course_Tools' directory.")
            print("Please ensure your notebook is saved within the main project folder.")
            return

        package_path = course_tools_dir / 'introdl'
        if not package_path.exists():
            print(f"❌ Error: The package directory was not found at: {package_path}")
            return

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(package_path)], 
                check=True, capture_output=True, text=True
            )
            print("\n✅ Installation successful!")
            print("\n‼️ IMPORTANT: Please restart the notebook kernel before proceeding.")
            
        except subprocess.CalledProcessError as e:
            print("\n❌ Installation failed. Please see the error below:")
            print(e.stderr)
    
    __COURSE_PACKAGE_CHECKED = True