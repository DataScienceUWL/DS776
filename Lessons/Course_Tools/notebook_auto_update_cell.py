# =============================================================================
# DS776 AUTO-UPDATE CELL
# Copy this cell to the TOP of every Lesson and Homework notebook
# It automatically checks and updates introdl when needed (usually <2 seconds)
# =============================================================================

import subprocess
import sys
from pathlib import Path

# Auto-detect notebook location and find the auto-update script
current_path = Path.cwd()
script_path = None

# Determine relative path to Course_Tools based on current location
if "Lessons" in str(current_path):
    # We're in a Lesson notebook: Lessons/LXX/ -> Lessons/Course_Tools/
    script_path = current_path.parent / "Course_Tools" / "auto_update_introdl.sh"
elif "Homework" in str(current_path):
    # We're in a Homework notebook: Homework/HW_XX/ -> Lessons/Course_Tools/
    script_path = current_path.parent.parent / "Lessons" / "Course_Tools" / "auto_update_introdl.sh"
else:
    # Try to find it by searching upwards
    for parent in [current_path] + list(current_path.parents):
        candidate = parent / "Lessons" / "Course_Tools" / "auto_update_introdl.sh"
        if candidate.exists():
            script_path = candidate
            break

if script_path and script_path.exists():
    try:
        # Run the auto-update script
        result = subprocess.run(
            ["bash", str(script_path)], 
            capture_output=True, 
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Display the output
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        
        # Check exit codes
        if result.returncode == 2:
            print("\n" + "="*60)
            print("🔄 KERNEL RESTART REQUIRED")
            print("The introdl package was updated.")
            print("Please RESTART THE KERNEL and run this cell again.")
            print("="*60)
        elif result.returncode != 0:
            print("\n" + "="*60)
            print("⚠️  AUTO-UPDATE ENCOUNTERED ISSUES")
            print("You may need to run the full Course_Setup.ipynb notebook.")
            print("="*60)
        else:
            print("\n✅ Ready to proceed with the lesson/homework!")
            
    except subprocess.TimeoutExpired:
        print("❌ Auto-update script timed out (network issues?)")
        print("📝 Try running the full Course_Setup.ipynb notebook if problems persist")
    except Exception as e:
        print(f"❌ Error running auto-update: {e}")
        print("📝 Try running the full Course_Setup.ipynb notebook")
        
else:
    print("❌ Could not find auto-update script")
    print(f"   Current location: {current_path}")
    print(f"   Looking for: Course_Tools/auto_update_introdl.sh")
    print("📝 Try running the full Course_Setup.ipynb notebook")