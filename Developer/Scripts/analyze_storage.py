#!/usr/bin/env python3
"""
Storage Analysis Script for DS776 Student Solutions
Runs each homework notebook and tracks storage usage.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import shutil
import time

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

def analyze_homework(hw_dir, hw_num):
    """
    Analyze storage for a single homework assignment.
    Creates local workspace and runs the notebook.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing HW{hw_num:02d}: {hw_dir}")
    print(f"{'='*60}")
    
    # Find the notebook to run
    notebooks = list(hw_dir.glob("*GRADE*.ipynb"))
    if not notebooks:
        notebooks = list(hw_dir.glob("*.ipynb"))
    
    if not notebooks:
        print(f"❌ No notebook found in {hw_dir}")
        return None
    
    notebook = notebooks[0]
    print(f"📓 Running: {notebook.name}")
    
    # Create home_workspace in the homework directory
    workspace = hw_dir / "home_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    
    data_path = workspace / "data"
    models_path = workspace / "models"
    cache_path = workspace / "downloads"
    
    for path in [data_path, models_path, cache_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Set environment for local workspace mode
    env = os.environ.copy()
    env['DS776_LOCAL_WORKSPACE'] = 'true'
    env['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU to avoid CUDA errors
    
    # Track initial sizes
    initial_sizes = {
        'data': get_dir_size(data_path),
        'models': get_dir_size(models_path),
        'cache': get_dir_size(cache_path),
    }
    
    # Run the notebook
    start_time = time.time()
    output_notebook = hw_dir / f"{notebook.stem}_output.ipynb"
    
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--ExecutePreprocessor.timeout=1200',  # 20 minutes timeout
        '--output', output_notebook.name,
        str(notebook)
    ]
    
    print("⏳ Executing notebook (this may take several minutes)...")
    result = subprocess.run(
        cmd,
        cwd=str(hw_dir),
        env=env,
        capture_output=True,
        text=True
    )
    
    execution_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"⚠️  Notebook execution failed (may be partial success)")
        print(f"   Error: {result.stderr[:500]}")
    else:
        print(f"✅ Notebook executed successfully")
    
    # Measure final sizes
    final_sizes = {
        'data': get_dir_size(data_path),
        'models': get_dir_size(models_path),
        'cache': get_dir_size(cache_path),
    }
    
    # Calculate peak workspace size
    total_size = sum(final_sizes.values())
    
    # Report results
    results = {
        'homework': f"HW{hw_num:02d}",
        'notebook': notebook.name,
        'execution_time': execution_time,
        'execution_status': 'success' if result.returncode == 0 else 'partial',
        'storage': {
            'data': {
                'initial': initial_sizes['data'],
                'final': final_sizes['data'],
                'formatted': format_size(final_sizes['data'])
            },
            'models': {
                'initial': initial_sizes['models'],
                'final': final_sizes['models'],
                'formatted': format_size(final_sizes['models'])
            },
            'cache': {
                'initial': initial_sizes['cache'],
                'final': final_sizes['cache'],
                'formatted': format_size(final_sizes['cache'])
            },
            'total': {
                'bytes': total_size,
                'formatted': format_size(total_size)
            }
        }
    }
    
    print(f"\n📊 Storage Summary for HW{hw_num:02d}:")
    print(f"   Data:   {results['storage']['data']['formatted']}")
    print(f"   Models: {results['storage']['models']['formatted']}")
    print(f"   Cache:  {results['storage']['cache']['formatted']}")
    print(f"   TOTAL:  {results['storage']['total']['formatted']}")
    print(f"   Time:   {execution_time:.1f} seconds")
    
    return results

def main():
    """Run storage analysis on all student homework solutions."""
    
    # Path to Ashley's solutions
    ashley_dir = Path("/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Ashley")
    
    if not ashley_dir.exists():
        print(f"❌ Ashley directory not found: {ashley_dir}")
        sys.exit(1)
    
    # Find all homework directories
    hw_dirs = sorted([d for d in ashley_dir.glob("HW*") if d.is_dir()])
    
    print(f"Found {len(hw_dirs)} homework directories")
    print(f"Starting analysis with local workspace mode...")
    
    all_results = []
    cumulative_storage = {
        'data': 0,
        'models': 0,
        'cache': 0,
        'total': 0
    }
    
    for hw_dir in hw_dirs:
        # Extract homework number
        hw_num = int(hw_dir.name[2:]) if hw_dir.name[2:].isdigit() else 0
        
        # Analyze this homework
        results = analyze_homework(hw_dir, hw_num)
        
        if results:
            all_results.append(results)
            
            # Track cumulative storage (assuming we keep all data)
            cumulative_storage['data'] += results['storage']['data']['final']
            cumulative_storage['models'] += results['storage']['models']['final']
            cumulative_storage['cache'] += results['storage']['cache']['final']
            cumulative_storage['total'] += results['storage']['total']['bytes']
    
    # Save results to JSON
    output_file = ashley_dir.parent / "storage_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\n📊 Cumulative Storage (if all homework data kept):")
    print(f"   Data:   {format_size(cumulative_storage['data'])}")
    print(f"   Models: {format_size(cumulative_storage['models'])}")
    print(f"   Cache:  {format_size(cumulative_storage['cache'])}")
    print(f"   TOTAL:  {format_size(cumulative_storage['total'])}")
    
    print(f"\n📊 Per-Homework Summary:")
    for result in all_results:
        print(f"   {result['homework']}: {result['storage']['total']['formatted']} "
              f"({result['execution_status']})")
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Check if 10GB is sufficient
    if cumulative_storage['total'] > 10 * 1024**3:  # 10GB in bytes
        print(f"\n⚠️  WARNING: Cumulative storage ({format_size(cumulative_storage['total'])}) "
              f"exceeds 10GB CoCalc limit!")
        print(f"   Storage management strategy needed!")
    else:
        print(f"\n✅ Cumulative storage fits within 10GB CoCalc limit")

if __name__ == "__main__":
    main()