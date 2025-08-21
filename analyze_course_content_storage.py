#!/usr/bin/env python3
"""
Script to analyze storage requirements for course content (Homework and Lessons folders).
These folders are part of the student's required storage.
"""

import os
import csv
from pathlib import Path


def get_folder_size(folder_path):
    """Calculate the total size of a folder in bytes."""
    total_size = 0
    if not folder_path.exists():
        return 0
    
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                # Skip symbolic links
                if not os.path.islink(filepath):
                    total_size += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass
    
    return total_size


def bytes_to_mb(bytes_value):
    """Convert bytes to megabytes."""
    return bytes_value / (1024 ** 2)


def analyze_homework_folders(base_path):
    """Analyze storage for all homework folders."""
    results = []
    homework_path = base_path / "Homework"
    
    if not homework_path.exists():
        print(f"Homework folder not found at {homework_path}")
        return results
    
    # Get all homework subdirectories
    hw_folders = sorted([d for d in homework_path.iterdir() if d.is_dir()])
    
    for hw_folder in hw_folders:
        print(f"Analyzing {hw_folder.name}...")
        
        # Analyze each file type
        ipynb_size = 0
        py_size = 0
        data_size = 0
        other_size = 0
        
        for dirpath, dirnames, filenames in os.walk(hw_folder):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                if not filepath.exists() or os.path.islink(filepath):
                    continue
                    
                size = os.path.getsize(filepath)
                
                # Categorize by file type
                if filename.endswith('.ipynb'):
                    ipynb_size += size
                elif filename.endswith('.py'):
                    py_size += size
                elif filename.endswith(('.csv', '.json', '.txt', '.mat', '.npy', '.npz', '.pkl', '.h5')):
                    data_size += size
                else:
                    other_size += size
        
        total_size = ipynb_size + py_size + data_size + other_size
        
        results.append({
            'Folder': hw_folder.name,
            'Notebooks_MB': round(bytes_to_mb(ipynb_size), 2),
            'Python_MB': round(bytes_to_mb(py_size), 2),
            'Data_MB': round(bytes_to_mb(data_size), 2),
            'Other_MB': round(bytes_to_mb(other_size), 2),
            'Total_MB': round(bytes_to_mb(total_size), 2)
        })
    
    return results


def analyze_lessons_folders(base_path):
    """Analyze storage for all lesson folders."""
    results = []
    lessons_path = base_path / "Lessons"
    
    if not lessons_path.exists():
        print(f"Lessons folder not found at {lessons_path}")
        return results
    
    # Get all lesson subdirectories
    lesson_folders = sorted([d for d in lessons_path.iterdir() if d.is_dir()])
    
    for lesson_folder in lesson_folders:
        print(f"Analyzing {lesson_folder.name}...")
        
        # Analyze each file type
        ipynb_size = 0
        py_size = 0
        data_size = 0
        media_size = 0
        other_size = 0
        
        for dirpath, dirnames, filenames in os.walk(lesson_folder):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                if not filepath.exists() or os.path.islink(filepath):
                    continue
                    
                size = os.path.getsize(filepath)
                
                # Categorize by file type
                if filename.endswith('.ipynb'):
                    ipynb_size += size
                elif filename.endswith('.py'):
                    py_size += size
                elif filename.endswith(('.csv', '.json', '.txt', '.mat', '.npy', '.npz', '.pkl', '.h5')):
                    data_size += size
                elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.mp4', '.avi', '.mov')):
                    media_size += size
                else:
                    other_size += size
        
        total_size = ipynb_size + py_size + data_size + media_size + other_size
        
        results.append({
            'Folder': lesson_folder.name,
            'Notebooks_MB': round(bytes_to_mb(ipynb_size), 2),
            'Python_MB': round(bytes_to_mb(py_size), 2),
            'Data_MB': round(bytes_to_mb(data_size), 2),
            'Media_MB': round(bytes_to_mb(media_size), 2),
            'Other_MB': round(bytes_to_mb(other_size), 2),
            'Total_MB': round(bytes_to_mb(total_size), 2)
        })
    
    return results


def write_results_to_csv(results, output_file, fieldnames):
    """Write results to CSV file."""
    if not results:
        print("No results to write")
        return
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to {output_file}")


def print_summary(results, title):
    """Print a summary of the storage analysis."""
    if not results:
        return
        
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print('=' * 80)
    
    # Determine column headers based on results
    if 'Media_MB' in results[0]:
        # Lessons format
        print(f"{'Folder':<30} {'Notebooks':>12} {'Python':>10} {'Data':>10} {'Media':>10} {'Other':>10} {'Total':>10}")
        print('-' * 92)
        for row in results:
            print(f"{row['Folder']:<30} {row['Notebooks_MB']:>10.2f}MB {row['Python_MB']:>8.2f}MB "
                  f"{row['Data_MB']:>8.2f}MB {row['Media_MB']:>8.2f}MB {row['Other_MB']:>8.2f}MB {row['Total_MB']:>8.2f}MB")
        
        # Calculate totals
        total_notebooks = sum(row['Notebooks_MB'] for row in results)
        total_python = sum(row['Python_MB'] for row in results)
        total_data = sum(row['Data_MB'] for row in results)
        total_media = sum(row['Media_MB'] for row in results)
        total_other = sum(row['Other_MB'] for row in results)
        grand_total = sum(row['Total_MB'] for row in results)
        
        print('-' * 92)
        print(f"{'TOTAL':<30} {total_notebooks:>10.2f}MB {total_python:>8.2f}MB "
              f"{total_data:>8.2f}MB {total_media:>8.2f}MB {total_other:>8.2f}MB {grand_total:>8.2f}MB")
    else:
        # Homework format
        print(f"{'Folder':<20} {'Notebooks':>12} {'Python':>10} {'Data':>10} {'Other':>10} {'Total':>10}")
        print('-' * 72)
        for row in results:
            print(f"{row['Folder']:<20} {row['Notebooks_MB']:>10.2f}MB {row['Python_MB']:>8.2f}MB "
                  f"{row['Data_MB']:>8.2f}MB {row['Other_MB']:>8.2f}MB {row['Total_MB']:>8.2f}MB")
        
        # Calculate totals
        total_notebooks = sum(row['Notebooks_MB'] for row in results)
        total_python = sum(row['Python_MB'] for row in results)
        total_data = sum(row['Data_MB'] for row in results)
        total_other = sum(row['Other_MB'] for row in results)
        grand_total = sum(row['Total_MB'] for row in results)
        
        print('-' * 72)
        print(f"{'TOTAL':<20} {total_notebooks:>10.2f}MB {total_python:>8.2f}MB "
              f"{total_data:>8.2f}MB {total_other:>8.2f}MB {grand_total:>8.2f}MB")
    
    print(f"\n{'Total Storage Required:':<25} {grand_total/1024:>8.2f} GB")
    
    # Find largest folder
    max_folder = max(results, key=lambda x: x['Total_MB'])
    print(f"{'Largest Folder:':<25} {max_folder['Folder']} ({max_folder['Total_MB']:.2f} MB)")


def main():
    """Main function to run the storage analysis."""
    # Set the base path to DS776 folder
    base_path = Path("/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776")
    
    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist")
        return
    
    print(f"Analyzing course content storage in: {base_path}")
    print("=" * 80)
    
    # Analyze Homework folders
    print("\nAnalyzing Homework folders...")
    hw_results = analyze_homework_folders(base_path)
    
    # Write Homework results to CSV
    hw_csv_file = base_path / "homework_content_storage.csv"
    hw_fieldnames = ['Folder', 'Notebooks_MB', 'Python_MB', 'Data_MB', 'Other_MB', 'Total_MB']
    write_results_to_csv(hw_results, hw_csv_file, hw_fieldnames)
    
    # Analyze Lessons folders
    print("\nAnalyzing Lessons folders...")
    lessons_results = analyze_lessons_folders(base_path)
    
    # Write Lessons results to CSV
    lessons_csv_file = base_path / "lessons_content_storage.csv"
    lessons_fieldnames = ['Folder', 'Notebooks_MB', 'Python_MB', 'Data_MB', 'Media_MB', 'Other_MB', 'Total_MB']
    write_results_to_csv(lessons_results, lessons_csv_file, lessons_fieldnames)
    
    # Print summaries
    print_summary(hw_results, "Homework Folders Storage Summary")
    print_summary(lessons_results, "Lessons Folders Storage Summary")
    
    # Overall summary
    hw_total = sum(row['Total_MB'] for row in hw_results)
    lessons_total = sum(row['Total_MB'] for row in lessons_results)
    
    print(f"\n{'=' * 80}")
    print("OVERALL COURSE CONTENT STORAGE")
    print('=' * 80)
    print(f"{'Total Homework Content:':<30} {hw_total:>10.2f} MB ({hw_total/1024:.2f} GB)")
    print(f"{'Total Lessons Content:':<30} {lessons_total:>10.2f} MB ({lessons_total/1024:.2f} GB)")
    print(f"{'Total Course Content:':<30} {hw_total + lessons_total:>10.2f} MB ({(hw_total + lessons_total)/1024:.2f} GB)")
    print(f"\nNote: This is the base storage students need before running any code or generating models.")


if __name__ == "__main__":
    main()