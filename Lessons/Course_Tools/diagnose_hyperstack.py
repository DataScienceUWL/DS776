#!/usr/bin/env python3
"""
Diagnostic script for Hyperstack environment to understand introdl installation issues.
Upload this to Hyperstack and run: python3 diagnose_hyperstack.py
"""

import os
import sys
from pathlib import Path
import site
import subprocess

print('='*60)
print('HYPERSTACK ENVIRONMENT DIAGNOSTICS')
print('='*60)

# Environment detection
print('\n1. ENVIRONMENT INDICATORS:')
print(f'  Hostname: {os.environ.get("HOSTNAME", "not set")}')
print(f'  User: {os.environ.get("USER", "not set")}')
print(f'  Home: {os.environ.get("HOME", "not set")}')
print(f'  Python: {sys.executable}')
print(f'  Python version: {sys.version}')
print(f'  /opt/hyperstack exists: {os.path.exists("/opt/hyperstack")}')
print(f'  /usr/local/cuda exists: {os.path.exists("/usr/local/cuda")}')
print(f'  /workspace exists: {os.path.exists("/workspace")}')

# Check for GPU/CUDA environment variables
cuda_vars = [k for k in os.environ if 'CUDA' in k or 'GPU' in k or 'NVIDIA' in k]
if cuda_vars:
    print(f'  CUDA/GPU vars: {cuda_vars}')

# Check for container/cloud indicators
container_files = [
    '/.dockerenv',
    '/run/.containerenv',
    '/var/run/secrets/kubernetes.io',
]
for cf in container_files:
    if os.path.exists(cf):
        print(f'  Container indicator: {cf} EXISTS')

print('\n2. INTRODL INSTALLATIONS FOUND:')
# All possible locations
locations = [
    Path(f'/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages'),
    Path(f'/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'),
    Path(f'/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages'),
    Path(f'/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'),
    Path('/opt/conda/lib/python3.10/site-packages'),
    Path('/opt/conda/lib/python3.11/site-packages'),
    Path('/opt/conda/lib/python3.9/site-packages'),
]

# Add site-packages paths
try:
    locations.extend([Path(p) for p in site.getsitepackages()])
except:
    pass

try:
    locations.append(Path(site.getusersitepackages()))
except:
    pass

# Add paths from sys.path
for p in sys.path:
    if 'packages' in p:
        locations.append(Path(p))

# Remove duplicates
locations = list(set(loc for loc in locations if loc.exists()))

# Check each location
found_any = False
all_installations = []

for loc in locations:
    introdl_path = loc / 'introdl'
    if introdl_path.exists():
        found_any = True
        all_installations.append(introdl_path)
        print(f'\n  Found at: {introdl_path}')

        # Check for old structure
        old_subdirs = ['utils', 'nlp', 'idlmam', 'visul']
        old_found = []
        for subdir in old_subdirs:
            subdir_path = introdl_path / subdir
            if subdir_path.exists():
                old_found.append(subdir)
                # Check if it has __init__.py
                if (subdir_path / '__init__.py').exists():
                    print(f'    ⚠️  {subdir}/__init__.py EXISTS (will cause import issues!)')

        if old_found:
            print(f'    ⚠️  OLD SUBDIRS PRESENT: {old_found}')

        # Check permissions
        try:
            test_file = introdl_path / '_test_write'
            test_file.touch()
            test_file.unlink()
            print(f'    ✅ Write permission: YES')
        except:
            print(f'    ❌ Write permission: NO (need sudo)')

        # Check for __init__.py
        init_file = introdl_path / '__init__.py'
        if init_file.exists():
            # Try to read version
            try:
                with open(init_file, 'r') as f:
                    for line in f:
                        if '__version__' in line:
                            print(f'    Version in file: {line.strip()}')
                            break
            except:
                pass

        # List contents
        try:
            contents = sorted([f.name for f in introdl_path.iterdir()])
            print(f'    Contents ({len(contents)} items): {contents[:15]}')
            if len(contents) > 15:
                print(f'    ... and {len(contents) - 15} more')
        except:
            print('    Could not list contents')

    # Also check for egg-info and dist-info
    for pattern in ['introdl*.egg-info', 'introdl*.dist-info']:
        for egg in loc.glob(pattern):
            print(f'  Metadata found: {egg}')

if not found_any:
    print('  No introdl installations found')
else:
    print(f'\n  TOTAL INSTALLATIONS: {len(all_installations)}')

print('\n3. PIP SHOW INTRODL:')
try:
    result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'introdl'],
                           capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f'  {line}')
    else:
        print('  Not installed via pip')
except Exception as e:
    print(f'  Error running pip show: {e}')

print('\n4. SOURCE DIRECTORY CHECK:')
# Try to find the course directory
possible_paths = [
    Path('/workspace/DS776'),
    Path.home() / 'DS776',
    Path('/content/DS776'),  # Colab-style
    Path('/mnt/DS776'),
    Path.cwd(),  # Current directory
]

source_found = False
for p in possible_paths:
    if p.exists():
        introdl_src = p / 'Lessons/Course_Tools/introdl'
        if introdl_src.exists():
            source_found = True
            print(f'  Course directory found at: {p}')
            print(f'    Source introdl at: {introdl_src}')

            # Check for build artifacts
            artifacts = ['build', 'dist', 'src/introdl.egg-info', '__pycache__']
            found_artifacts = []
            for artifact in artifacts:
                artifact_path = introdl_src / artifact
                if artifact_path.exists():
                    found_artifacts.append(artifact)

            if found_artifacts:
                print(f'    ⚠️  Build artifacts present: {found_artifacts}')

            # Check source version
            init_path = introdl_src / 'src' / 'introdl' / '__init__.py'
            if init_path.exists():
                try:
                    with open(init_path, 'r') as f:
                        for line in f:
                            if '__version__' in line:
                                print(f'    Source version: {line.strip()}')
                                break
                except:
                    pass
            break

if not source_found:
    print('  No source directory found in standard locations')
    print('  Current working directory:', Path.cwd())

print('\n5. PYTHON IMPORT TEST:')
# Try to import and see what happens
try:
    import introdl
    print(f'  ✅ introdl imports successfully')
    print(f'  Version: {getattr(introdl, "__version__", "unknown")}')
    print(f'  Location: {introdl.__file__}')

    # Check for old submodules
    old_modules = ['utils', 'nlp', 'idlmam', 'visul']
    for mod in old_modules:
        if hasattr(introdl, mod):
            submod = getattr(introdl, mod)
            if hasattr(submod, '__file__'):
                print(f'  ⚠️  OLD MODULE {mod} exists at: {submod.__file__}')
except ImportError as e:
    print(f'  ❌ Import failed: {e}')
except Exception as e:
    print(f'  ❌ Error during import: {e}')

print('\n6. SYS.MODULES CHECK:')
# Check what's in sys.modules
introdl_modules = [m for m in sys.modules if m.startswith('introdl')]
if introdl_modules:
    print(f'  introdl modules in sys.modules: {introdl_modules}')
else:
    print('  No introdl modules in sys.modules')

print('\n7. RECOMMENDED ACTIONS:')
if found_any and old_found:
    print('  ⚠️  OLD PACKAGE STRUCTURE DETECTED!')
    print('  The following commands should fix it:')
    for install_path in all_installations:
        print(f'\n  # Remove old structure from {install_path}:')
        for subdir in old_found:
            print(f'  sudo rm -rf {install_path / subdir}')
    print('\n  # Then reinstall:')
    print('  pip uninstall -y introdl')
    print('  pip install ./Lessons/Course_Tools/introdl --no-cache-dir')

print('\n' + '='*60)
print('END OF DIAGNOSTICS')
print('='*60)