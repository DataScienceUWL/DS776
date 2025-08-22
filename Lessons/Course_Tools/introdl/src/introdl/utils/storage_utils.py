"""Storage management utilities for DS776 course."""
import os
import shutil
from pathlib import Path
import time
from datetime import datetime, timedelta
import pandas as pd


def get_folder_size(path):
    """Calculate total size of a folder in bytes."""
    total = 0
    if path.exists():
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                if filepath.exists():
                    total += filepath.stat().st_size
    return total


def format_size(bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def get_comprehensive_storage_report():
    """
    Get a comprehensive storage report for the entire DS776 course.
    
    Returns:
        dict: Storage information organized by category
    """
    from pathlib import Path
    import os
    
    # Get course root
    course_root = Path(os.environ.get('DS776_ROOT_DIR', Path.home()))
    home_workspace = course_root / "home_workspace"
    cs_workspace = Path.home() / "cs_workspace"
    
    report = {
        'summary': {},
        'workspaces': {},
        'lessons': {},
        'homework': {},
        'solutions': {},
        'cache': {},
        'recommendations': []
    }
    
    # Check home_workspace
    if home_workspace.exists():
        report['workspaces']['home_workspace'] = {
            'path': home_workspace,
            'total': get_folder_size(home_workspace),
            'data': get_folder_size(home_workspace / "data"),
            'downloads': get_folder_size(home_workspace / "downloads"),
            'models': get_folder_size(home_workspace / "models"),
        }
    
    # Check cs_workspace (compute server)
    if cs_workspace.exists():
        report['workspaces']['cs_workspace'] = {
            'path': cs_workspace,
            'total': get_folder_size(cs_workspace),
            'downloads': get_folder_size(cs_workspace / "downloads"),
        }
    
    # Check Lessons
    lessons_dir = course_root / "Lessons"
    if lessons_dir.exists():
        total_lessons = 0
        for lesson_dir in sorted(lessons_dir.glob("Lesson_*")):
            if lesson_dir.is_dir():
                lesson_size = get_folder_size(lesson_dir)
                models_dir = None
                models_size = 0
                
                # Find any Lesson_X_models directory
                for models_path in lesson_dir.glob("Lesson_*_models"):
                    models_dir = models_path
                    models_size = get_folder_size(models_path)
                    break
                
                report['lessons'][lesson_dir.name] = {
                    'path': lesson_dir,
                    'total': lesson_size,
                    'models': models_size,
                    'models_path': models_dir,
                    'other': lesson_size - models_size
                }
                total_lessons += lesson_size
        report['summary']['lessons_total'] = total_lessons
    
    # Check Homework
    homework_dir = course_root / "Homework"
    if homework_dir.exists():
        total_homework = 0
        for hw_dir in sorted(homework_dir.glob("Homework_*")):
            if hw_dir.is_dir():
                hw_size = get_folder_size(hw_dir)
                models_dir = None
                models_size = 0
                
                # Find any Homework_X_models directory
                for models_path in hw_dir.glob("Homework_*_models"):
                    models_dir = models_path
                    models_size = get_folder_size(models_path)
                    break
                
                report['homework'][hw_dir.name] = {
                    'path': hw_dir,
                    'total': hw_size,
                    'models': models_size,
                    'models_path': models_dir,
                    'other': hw_size - models_size
                }
                total_homework += hw_size
        report['summary']['homework_total'] = total_homework
    
    # Check Solutions (only if exists)
    solutions_dir = course_root / "Solutions"
    if solutions_dir.exists():
        report['summary']['solutions_total'] = get_folder_size(solutions_dir)
    else:
        report['summary']['solutions_total'] = 0
    
    # Calculate cache sizes and what can be freed
    cache_locations = []
    
    # Home workspace cache
    if home_workspace.exists():
        hw_downloads = home_workspace / "downloads"
        if hw_downloads.exists():
            cache_locations.append(('home_workspace/downloads', hw_downloads))
        
        hw_data = home_workspace / "data"
        if hw_data.exists():
            cache_locations.append(('home_workspace/data', hw_data))
    
    # CS workspace cache
    if cs_workspace.exists():
        cs_downloads = cs_workspace / "downloads"
        if cs_downloads.exists():
            cache_locations.append(('cs_workspace/downloads', cs_downloads))
    
    total_cache = 0
    old_cache = 0
    for name, path in cache_locations:
        size = get_folder_size(path)
        old_size = get_old_files_size(path, days=7)
        report['cache'][name] = {
            'path': path,
            'total': size,
            'old_files': old_size
        }
        total_cache += size
        old_cache += old_size
    
    report['summary']['cache_total'] = total_cache
    report['summary']['old_cache'] = old_cache
    
    # Calculate storage by location type
    # Shared storage (synced in CoCalc base, limited to 10GB)
    shared_storage = sum([
        report['summary'].get('lessons_total', 0),
        report['summary'].get('homework_total', 0),
        report['summary'].get('solutions_total', 0),
        report['workspaces'].get('home_workspace', {}).get('total', 0),
    ])
    report['summary']['shared_storage'] = shared_storage
    
    # Compute server local storage (not synced, ~50GB available)
    compute_storage = report['workspaces'].get('cs_workspace', {}).get('total', 0)
    report['summary']['compute_storage'] = compute_storage
    
    # Grand total
    grand_total = shared_storage + compute_storage
    report['summary']['grand_total'] = grand_total
    
    # Add recommendations based on environment
    from ..utils import detect_jupyter_environment
    environment = detect_jupyter_environment()
    is_local = (environment == "Local Development")
    
    if old_cache > 1e9:  # More than 1GB of old cache
        report['recommendations'].append(
            f"🧹 Clean old cache files: {format_size(old_cache)} can be freed"
        )
    
    # Only warn about limits in CoCalc environments
    if not is_local and shared_storage > 8e9:  # More than 8GB shared
        report['recommendations'].append(
            "⚠️ Approaching 10GB shared storage limit - consider cleanup"
        )
    elif not is_local and shared_storage > 6e9:
        report['recommendations'].append(
            f"🟡 Using {format_size(shared_storage)} of 10GB shared storage"
        )
    
    # Find largest lesson/homework models
    largest_models = []
    for lesson_name, lesson_info in report['lessons'].items():
        if lesson_info['models'] > 0:
            largest_models.append((lesson_name, lesson_info['models']))
    for hw_name, hw_info in report['homework'].items():
        if hw_info['models'] > 0:
            largest_models.append((hw_name, hw_info['models']))
    
    largest_models.sort(key=lambda x: x[1], reverse=True)
    if largest_models and largest_models[0][1] > 500e6:  # More than 500MB
        report['recommendations'].append(
            f"💾 Largest models: {largest_models[0][0]} ({format_size(largest_models[0][1])})"
        )
    
    return report


def display_storage_report(report=None):
    """
    Display a comprehensive storage report.
    
    Args:
        report: Storage report dict from get_comprehensive_storage_report()
                If None, generates a new report
    """
    if report is None:
        report = get_comprehensive_storage_report()
    
    # Detect environment
    from ..utils import detect_jupyter_environment
    environment = detect_jupyter_environment()
    is_local = (environment == "vscode" or environment == "unknown")
    is_compute_server = (environment == "cocalc_compute_server")
    is_cocalc_base = (environment == "cocalc")
    
    print("=" * 60)
    print("📊 DS776 COMPREHENSIVE STORAGE REPORT")
    print("=" * 60)
    print(f"📍 Environment: {environment}")
    
    # Get storage breakdowns
    lessons_total = report['summary'].get('lessons_total', 0)
    homework_total = report['summary'].get('homework_total', 0)
    solutions_total = report['summary'].get('solutions_total', 0)
    
    # Calculate cache/downloads separately for shared and compute
    home_workspace_total = report['workspaces'].get('home_workspace', {}).get('total', 0)
    home_downloads = report['workspaces'].get('home_workspace', {}).get('downloads', 0)
    home_data = report['workspaces'].get('home_workspace', {}).get('data', 0)
    shared_cache = home_downloads + home_data
    
    cs_workspace_total = report['workspaces'].get('cs_workspace', {}).get('total', 0)
    compute_cache = report['workspaces'].get('cs_workspace', {}).get('downloads', 0)
    
    # Calculate shared storage (without home_workspace as a whole)
    shared_storage = lessons_total + homework_total + solutions_total + shared_cache
    
    total = shared_storage + compute_cache
    
    # SHARED STORAGE SECTION
    print("\n📦 SHARED STORAGE (Common to CoCalc & Compute Server)")
    print("-" * 40)
    
    if is_local:
        print(f"Total: {format_size(shared_storage)} (no limits in local development)")
    else:
        # Show limit status
        if shared_storage > 8e9:
            status = "⚠️ WARNING: Approaching 10GB limit!"
            remaining = f"{format_size(10e9 - shared_storage)} remaining"
        elif shared_storage > 5e9:
            status = "🟡 Caution"
            remaining = f"{format_size(10e9 - shared_storage)} remaining of 10GB"
        else:
            status = "🟢 Healthy"
            remaining = f"{format_size(10e9 - shared_storage)} remaining of 10GB"
        
        print(f"Total: {format_size(shared_storage)} - {status}")
        print(f"  {remaining}")
    
    # Calculate model totals for lessons and homework
    lessons_models_total = sum(info.get('models', 0) for info in report['lessons'].values())
    lessons_other_total = lessons_total - lessons_models_total
    
    homework_models_total = sum(info.get('models', 0) for info in report['homework'].values())
    homework_other_total = homework_total - homework_models_total
    
    # Breakdown of shared storage
    print("\n  Breakdown:")
    
    # Lessons with sub-breakdown
    if lessons_total > 0:
        print(f"  • Lessons:          {format_size(lessons_total):>10}")
        if lessons_models_total > 0 or lessons_other_total > 0:
            print(f"      - Trained models:      {format_size(lessons_models_total):>10} (can be deleted)")
            print(f"      - Notebooks & scripts: {format_size(lessons_other_total):>10}")
    
    # Homework with sub-breakdown
    if homework_total > 0:
        print(f"  • Homework:         {format_size(homework_total):>10}")
        if homework_models_total > 0 or homework_other_total > 0:
            print(f"      - Your trained models: {format_size(homework_models_total):>10} (can be zipped/deleted)")
            print(f"      - Notebooks & scripts: {format_size(homework_other_total):>10}")
    
    # Solutions (no breakdown needed)
    if solutions_total > 0:
        print(f"  • Solutions:        {format_size(solutions_total):>10}")
    
    # Cache
    if shared_cache > 0:
        print(f"  • Downloaded Data/Cache: {format_size(shared_cache):>10} (pretrained models)")
    
    # Show total reclaimable space
    total_reclaimable = lessons_models_total + homework_models_total + shared_cache
    if total_reclaimable > 100000:  # Only show if meaningful amount
        print(f"\n  💡 Reclaimable Space: {format_size(total_reclaimable):>10}")
        print(f"     (Delete trained models + clear cache)")
    
    # COMPUTE SERVER STORAGE SECTION
    if compute_cache > 0:
        print("\n💾 COMPUTE SERVER STORAGE (Local to compute server only)")
        print("-" * 40)
        print(f"Total: {format_size(compute_cache)}")
        print(f"  • Downloaded Data/Cache: {format_size(compute_cache):>10}")
        print(f"  📍 Location: cs_workspace/downloads")
        print(f"  💡 ~50GB available, not synced to base CoCalc")
    
    print(f"\n📊 GRAND TOTAL: {format_size(total)}")
    
    
    # Top 5 largest items
    print("\n🔝 TOP 5 LARGEST ITEMS")
    print("-" * 40)
    
    all_items = []
    for lesson_name, lesson_info in report['lessons'].items():
        all_items.append((lesson_name, lesson_info['total']))
    for hw_name, hw_info in report['homework'].items():
        all_items.append((hw_name, hw_info['total']))
    
    all_items.sort(key=lambda x: x[1], reverse=True)
    for i, (name, size) in enumerate(all_items[:5], 1):
        print(f"{i}. {name:25} {format_size(size):>10}")
    
    # Cleanup opportunities
    print("\n🧹 CLEANUP OPPORTUNITIES")
    print("-" * 40)
    
    # Calculate totals for each type
    lessons_models_total = sum(info.get('models', 0) for info in report['lessons'].values())
    homework_models_total = sum(info.get('models', 0) for info in report['homework'].values())
    old_cache = report['summary'].get('old_cache', 0)
    
    # Count model folders
    lesson_model_count = sum(1 for info in report['lessons'].values() if info.get('models', 0) > 0)
    homework_model_count = sum(1 for info in report['homework'].values() if info.get('models', 0) > 0)
    
    if lessons_models_total > 0:
        print(f"• Lesson trained models ({lesson_model_count} folders): {format_size(lessons_models_total)}")
        print(f"    Use: Homework utility notebooks → Delete Lesson Models")
    
    if homework_models_total > 0:
        print(f"• Homework trained models ({homework_model_count} folders): {format_size(homework_models_total)}")
        print(f"    Use: Homework utility notebooks → Zip Models")
    
    if old_cache > 0:
        print(f"• Old pretrained models (>7 days): {format_size(old_cache)}")
        print(f"    Use: Homework utility notebooks → Clean Cache")
    
    total_reclaimable = lessons_models_total + homework_models_total + old_cache
    if total_reclaimable > 0:
        print(f"\n💡 Total reclaimable space: {format_size(total_reclaimable)}")
    
    # Recommendations
    if report['recommendations']:
        print("\n📋 RECOMMENDATIONS")
        print("-" * 40)
        for rec in report['recommendations']:
            print(rec)
    
    print("\n" + "=" * 60)


def get_old_files_size(path, days=7):
    """Calculate total size of files older than specified days."""
    if not path.exists():
        return 0
    
    cutoff_time = time.time() - (days * 24 * 3600)
    total_size = 0
    
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = Path(root) / file
            try:
                if filepath.stat().st_mtime < cutoff_time:
                    total_size += filepath.stat().st_size
            except:
                pass
    
    return total_size


def cleanup_old_cache(days_old=7, dry_run=True):
    """
    Remove cached models and data files older than specified days.
    
    Args:
        days_old: Remove files older than this many days
        dry_run: If True, only show what would be deleted
        
    Returns:
        int: Total bytes freed (or would be freed if dry_run)
    """
    from pathlib import Path
    import os
    
    # Get all cache locations
    course_root = Path(os.environ.get('DS776_ROOT_DIR', Path.home()))
    home_workspace = course_root / "home_workspace"
    cs_workspace = Path.home() / "cs_workspace"
    
    cache_locations = []
    
    # Add home_workspace locations
    if home_workspace.exists():
        if (home_workspace / "downloads").exists():
            cache_locations.append(("home_workspace/downloads", home_workspace / "downloads"))
        if (home_workspace / "data").exists():
            cache_locations.append(("home_workspace/data", home_workspace / "data"))
    
    # Add cs_workspace locations
    if cs_workspace.exists():
        if (cs_workspace / "downloads").exists():
            cache_locations.append(("cs_workspace/downloads", cs_workspace / "downloads"))
    
    if not cache_locations:
        print("ℹ️ No cache directories found.")
        return 0
    
    print("\n🧹 CACHE CLEANUP")
    print("-" * 40)
    print(f"{'DRY RUN - ' if dry_run else ''}Removing files older than {days_old} days")
    print(f"⚠️ NOTE: This does NOT delete your trained models in Lesson/Homework folders!")
    print("   Only cached downloads and datasets are affected.\n")
    
    cutoff_time = time.time() - (days_old * 24 * 3600)
    files_to_delete = []
    total_size = 0
    
    # Find old files in each location
    for location_name, location_path in cache_locations:
        location_files = []
        location_size = 0
        
        for root, dirs, files in os.walk(location_path):
            for file in files:
                filepath = Path(root) / file
                try:
                    if filepath.stat().st_mtime < cutoff_time:
                        size = filepath.stat().st_size
                        location_files.append((filepath, size))
                        location_size += size
                        total_size += size
                except:
                    pass
        
        if location_files:
            files_to_delete.extend(location_files)
            print(f"📁 {location_name}: {len(location_files)} files, {format_size(location_size)}")
    
    if not files_to_delete:
        print("✅ No old cache files found. Everything is recent!")
        return 0
    
    print(f"\n📊 Total: {len(files_to_delete)} files, {format_size(total_size)}")
    
    if dry_run:
        print("\n👀 DRY RUN MODE - No files were deleted")
        print("   Set dry_run=False to actually delete files")
        
        # Show a few example files
        if len(files_to_delete) > 0:
            print("\n   Example files that would be deleted:")
            for filepath, size in files_to_delete[:3]:
                print(f"   • {filepath.name} ({format_size(size)})")
            if len(files_to_delete) > 3:
                print(f"   ... and {len(files_to_delete) - 3} more files")
    else:
        print("\n🗑️ Deleting old cache files...")
        deleted_count = 0
        failed_count = 0
        
        for filepath, _ in files_to_delete:
            try:
                filepath.unlink()
                deleted_count += 1
            except Exception as e:
                failed_count += 1
        
        print(f"\n✅ Cleanup complete!")
        print(f"   • Deleted: {deleted_count} files")
        if failed_count > 0:
            print(f"   • Failed: {failed_count} files")
        print(f"   • Freed: {format_size(total_size)}")
    
    return total_size


def delete_lesson_or_homework_models(target_dir_name, confirm=False):
    """
    Delete models folder from a specific Lesson or Homework directory.
    
    Args:
        target_dir_name: Name of the target directory (e.g., "Lesson_2" or "Homework_3")
        confirm: If True, actually delete the folder
        
    Returns:
        int: Bytes freed (or would be freed)
    """
    from pathlib import Path
    import os
    
    course_root = Path(os.environ.get('DS776_ROOT_DIR', Path.home()))
    
    # Determine if it's a lesson or homework
    if target_dir_name.startswith("Lesson_"):
        parent_dir = course_root / "Lessons"
        # Find the full directory name (might have description after number)
        candidates = list(parent_dir.glob(f"{target_dir_name}*"))
    elif target_dir_name.startswith("Homework_"):
        parent_dir = course_root / "Homework"
        candidates = [parent_dir / target_dir_name]
    else:
        print(f"❌ Invalid directory name: {target_dir_name}")
        print("   Must start with 'Lesson_' or 'Homework_'")
        return 0
    
    if not candidates or not candidates[0].exists():
        print(f"❌ Directory not found: {target_dir_name}")
        return 0
    
    target_dir = candidates[0]
    
    # Find models directory
    models_dirs = list(target_dir.glob("*_models"))
    
    if not models_dirs:
        print(f"ℹ️ No models folder found in {target_dir.name}")
        return 0
    
    models_dir = models_dirs[0]
    size = get_folder_size(models_dir)
    
    print(f"\n🗂️ MODELS FOLDER DELETION")
    print("-" * 40)
    print(f"Target: {models_dir.name}")
    print(f"Location: {models_dir.parent.name}")
    print(f"Size: {format_size(size)}")
    
    if not confirm:
        print("\n⚠️ This will permanently delete the models folder!")
        print("   Set confirm=True to proceed with deletion")
        return 0
    
    print("\n🗑️ Deleting models folder...")
    try:
        shutil.rmtree(models_dir)
        print(f"✅ Successfully deleted {models_dir.name}")
        print(f"   Freed: {format_size(size)}")
        return size
    except Exception as e:
        print(f"❌ Error deleting folder: {e}")
        return 0


def delete_current_lesson_models(confirm=False):
    """
    Delete the models folder for the lesson corresponding to current homework.
    Automatically detects which lesson based on current directory.
    
    Args:
        confirm: If True, actually delete the folder
        
    Returns:
        int: Bytes freed (or would be freed)
    """
    from pathlib import Path
    
    # Get current directory
    cwd = Path.cwd()
    
    # Extract homework number
    if not cwd.name.startswith("Homework_"):
        print("❌ This function must be run from a Homework directory")
        return 0
    
    # Extract homework number
    parts = cwd.name.split("_")
    if len(parts) < 2:
        print("❌ Could not determine homework number")
        return 0
    
    hw_num = parts[1]  # Keep as-is (e.g., "01")
    
    # Find corresponding lesson - use same format
    lesson_name = f"Lesson_{hw_num}"
    
    print(f"🔍 Looking for {lesson_name} models...")
    return delete_lesson_or_homework_models(lesson_name, confirm=confirm)


def export_homework_html_interactive(notebook_name=None):
    """
    Export a notebook to HTML. If notebook_name not provided, shows list of available notebooks.
    
    Args:
        notebook_name: Name of notebook to export (e.g., "Homework_01_Classify_Spiral_Points.ipynb")
                      If None, lists available notebooks
    
    Returns:
        Path to the created HTML file, or None if failed
    """
    from pathlib import Path
    
    # Find all notebooks in current directory
    cwd = Path.cwd()
    notebooks = sorted([f for f in cwd.glob("*.ipynb") 
                       if 'utilities' not in f.name.lower() 
                       and 'utils' not in f.name.lower()
                       and 'clean' not in f.name.lower()])
    
    if not notebooks:
        print("❌ No notebooks found in current directory")
        return None
    
    # If no notebook specified, show available options
    if notebook_name is None:
        print("\n📚 AVAILABLE NOTEBOOKS FOR EXPORT")
        print("-" * 40)
        for nb in notebooks:
            print(f"• {nb.name}")
        print("\n💡 To export, call this function with the notebook name:")
        print(f'   export_homework_html_interactive("{notebooks[0].name}")')
        return None
    
    # Find the specified notebook
    selected_notebook = None
    for nb in notebooks:
        if nb.name == notebook_name:
            selected_notebook = nb
            break
    
    if selected_notebook is None:
        print(f"❌ Notebook not found: {notebook_name}")
        print("\n📚 Available notebooks:")
        for nb in notebooks:
            print(f"• {nb.name}")
        return None
    
    # Generate output filename (same name but .html)
    output_name = selected_notebook.stem + ".html"
    
    print(f"\n📄 EXPORTING TO HTML")
    print("-" * 40)
    print(f"Source: {selected_notebook.name}")
    print(f"Output: {output_name}")
    
    # Use the existing convert_nb_to_html function
    from ..utils import convert_nb_to_html
    
    try:
        convert_nb_to_html(
            output_filename=output_name,
            notebook_path=selected_notebook,
            template="lab"
        )
        
        output_path = cwd / output_name
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"\n✅ Export successful!")
            print(f"   File: {output_name}")
            print(f"   Size: {format_size(size)}")
            print(f"\n📤 Ready to upload to Canvas!")
            return output_path
        else:
            print("❌ Export may have failed - output file not found")
            return None
            
    except Exception as e:
        print(f"❌ Error during export: {e}")
        print("\n💡 Alternative: Use File → Download as → HTML from Jupyter menu")
        return None


def zip_homework_models(hw_num=None, delete_after=False):
    """
    Zip the homework models folder for download.
    
    Args:
        hw_num: Homework number (e.g., '01', '02'). If None, auto-detect.
        delete_after: If True, delete the original folder after zipping
    
    Returns:
        Path to zip file if successful, None otherwise
    """
    import zipfile
    from datetime import datetime
    
    # Auto-detect homework number if not provided
    if hw_num is None:
        current_dir = Path.cwd()
        if 'Homework_' in str(current_dir):
            hw_num = current_dir.name.split('_')[1]
        else:
            print("❌ Could not auto-detect homework number")
            print("   Please provide hw_num parameter")
            return None
    
    # Find the models folder - try both formats
    models_folder = Path.cwd() / f"Homework_{hw_num}_models"
    
    # If not found, try without zero padding
    if not models_folder.exists():
        hw_num_int = str(int(hw_num)) if hw_num.isdigit() else hw_num
        models_folder = Path.cwd() / f"Homework_{hw_num_int}_models"
    
    if not models_folder.exists():
        print(f"❌ Models folder not found")
        print(f"   Looked for: Homework_{hw_num}_models or Homework_{hw_num_int}_models")
        print("   No models to zip")
        return None
    
    # Get folder size
    folder_size = get_folder_size(models_folder)
    
    # Create zip filename with timestamp using consistent naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"Homework_{hw_num}_models_{timestamp}.zip"
    zip_path = Path.cwd() / zip_name
    
    print(f"\n📦 ZIP MODELS FOR DOWNLOAD")
    print("-" * 40)
    print(f"📁 Source: {models_folder.name}")
    print(f"📏 Size: {format_size(folder_size)}")
    print(f"🗜️ Target: {zip_name}")
    print("\n⏳ Creating zip file...")
    
    try:
        # Create zip file
        file_count = 0
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the models folder
            for root, dirs, files in os.walk(models_folder):
                for file in files:
                    file_path = Path(root) / file
                    # Add file to zip with relative path from models folder
                    # This ensures files are zipped with proper structure
                    arcname = models_folder.name + '/' + str(file_path.relative_to(models_folder))
                    zipf.write(file_path, arcname)
                    file_count += 1
            
            # If no files found, add the folder structure at least
            if file_count == 0:
                # Add an empty marker file to preserve folder structure
                zipf.writestr(f"{models_folder.name}/.empty", "")
                print("   ⚠️ Note: Models folder is empty")
        
        # Get zip file size
        zip_size = zip_path.stat().st_size
        compression_ratio = (1 - zip_size / folder_size) * 100 if folder_size > 0 else 0
        
        print(f"\n✅ SUCCESS")
        print(f"📦 Created: {zip_name}")
        print(f"📊 Files: {file_count} files compressed")
        print(f"📏 Size: {format_size(zip_size)} ({compression_ratio:.1f}% compression)")
        
        # Delete original if requested
        if delete_after:
            print(f"\n🗑️ DELETING ORIGINAL")
            print(f"Removing: {models_folder.name}")
            import shutil
            shutil.rmtree(models_folder)
            print(f"✅ Deleted successfully")
            print(f"💾 Space freed: {format_size(folder_size)}")
        else:
            print(f"\n💡 TIP: Original folder kept")
            print(f"   To also delete it, use: delete_after=True")
        
        return zip_path
        
    except Exception as e:
        print(f"❌ Error creating zip: {e}")
        # Clean up partial zip if it exists
        if zip_path.exists():
            zip_path.unlink()
        return None