#!/usr/bin/env python3
"""
Diagnostic script to troubleshoot introdl installation issues on Hyperstack compute servers
"""

import subprocess
import sys
import os
from pathlib import Path
import site
import json

print("=" * 70)
print("INTRODL INSTALLATION DIAGNOSTIC")
print("=" * 70)

# 1. Python environment info
print("\n1. PYTHON ENVIRONMENT:")
print(f"   Python executable: {sys.executable}")
print(f"   Python version: {sys.version}")
print(f"   Python prefix: {sys.prefix}")
print(f"   Virtual env: {sys.prefix != sys.base_prefix}")

# 2. Site packages locations
print("\n2. SITE-PACKAGES LOCATIONS:")
print(f"   User site: {site.getusersitepackages()}")
print(f"   Site packages: {site.getsitepackages()}")

# 3. sys.path
print("\n3. PYTHON PATH (sys.path):")
for i, p in enumerate(sys.path[:10]):  # First 10 entries
    print(f"   [{i}] {p}")
if len(sys.path) > 10:
    print(f"   ... and {len(sys.path)-10} more")

# 4. Look for ALL introdl installations
print("\n4. SEARCHING FOR ALL INTRODL INSTALLATIONS:")
found_installations = []

# Search in all possible locations
search_paths = [
    Path(site.getusersitepackages()),
    Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    Path(sys.prefix) / "local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
]

# Add conda environments if present
if 'conda' in sys.executable.lower() or 'CONDA_PREFIX' in os.environ:
    conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
    search_paths.append(Path(conda_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")

# Add any site-packages from sys.path
for p in sys.path:
    if 'site-packages' in p:
        search_paths.append(Path(p))

# Remove duplicates and non-existent paths
search_paths = list(set(p for p in search_paths if p and p.exists()))

for location in search_paths:
    introdl_dir = location / "introdl"
    if introdl_dir.exists():
        init_file = introdl_dir / "__init__.py"
        version = "unknown"
        if init_file.exists():
            try:
                with open(init_file, 'r') as f:
                    for line in f:
                        if '__version__' in line and '=' in line:
                            import re
                            match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', line)
                            if match:
                                version = match.group(1)
                                break
            except:
                pass

        found_installations.append({
            'location': str(location),
            'version': version,
            'init_exists': init_file.exists(),
            'init_size': init_file.stat().st_size if init_file.exists() else 0
        })
        print(f"   ✓ Found in: {location}")
        print(f"     Version: {version}")
        print(f"     __init__.py size: {init_file.stat().st_size if init_file.exists() else 0} bytes")

if not found_installations:
    print("   ✗ No introdl installations found")

# 5. pip show output
print("\n5. PIP SHOW INTRODL:")
try:
    result = subprocess.run([sys.executable, "-m", "pip", "show", "introdl"],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            if line.startswith(('Version:', 'Location:', 'Name:')):
                print(f"   {line}")
    else:
        print("   ✗ Package not found by pip")
except Exception as e:
    print(f"   ✗ Error running pip show: {e}")

# 6. pip list | grep introdl
print("\n6. PIP LIST (introdl entries):")
try:
    result = subprocess.run([sys.executable, "-m", "pip", "list"],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'introdl' in line.lower():
                print(f"   {line}")
    else:
        print("   ✗ Error running pip list")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 7. Import test
print("\n7. IMPORT TEST:")
try:
    import introdl
    print(f"   ✓ Import successful")
    print(f"   Version: {introdl.__version__}")
    print(f"   File location: {introdl.__file__}")

    # Check if train_network exists
    if hasattr(introdl, 'train_network'):
        print(f"   ✓ train_network found in main namespace")
    else:
        print(f"   ✗ train_network NOT in main namespace")

    # Check in idlmam
    try:
        from introdl.idlmam import train_network
        print(f"   ✓ train_network imported from idlmam")

        # Check for new parameters
        import inspect
        sig = inspect.signature(train_network)
        params = list(sig.parameters.keys())

        new_params = ['save_last', 'resume_last', 'total_epochs']
        for param in new_params:
            if param in params:
                print(f"   ✓ Parameter '{param}' found")
            else:
                print(f"   ✗ Parameter '{param}' NOT found")

    except Exception as e:
        print(f"   ✗ Error importing train_network: {e}")

except ImportError as e:
    print(f"   ✗ Import failed: {e}")
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")

# 8. Check environment variables
print("\n8. RELEVANT ENVIRONMENT VARIABLES:")
for var in ['PYTHONPATH', 'PIP_PREFIX', 'PIP_TARGET', 'CONDA_PREFIX', 'VIRTUAL_ENV']:
    value = os.environ.get(var)
    if value:
        print(f"   {var}={value}")

# 9. Check if we're on a compute server
print("\n9. COMPUTE SERVER CHECK:")
hostname = os.environ.get('HOSTNAME', 'unknown')
print(f"   Hostname: {hostname}")

# Check if /home/user/.local exists (common on compute servers)
home_local = Path.home() / ".local"
if home_local.exists():
    print(f"   ✓ ~/.local exists")
    local_site = home_local / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if local_site.exists():
        print(f"   ✓ ~/.local site-packages exists")
        local_introdl = local_site / "introdl"
        if local_introdl.exists():
            print(f"   ✓ introdl found in ~/.local")
            # Check version
            init_file = local_introdl / "__init__.py"
            if init_file.exists():
                try:
                    with open(init_file, 'r') as f:
                        for line in f:
                            if '__version__' in line and '=' in line:
                                import re
                                match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', line)
                                if match:
                                    version = match.group(1)
                                    print(f"     Version in ~/.local: {version}")
                                    break
                except:
                    pass

# 10. Check pip cache
print("\n10. PIP CACHE CHECK:")
try:
    result = subprocess.run([sys.executable, "-m", "pip", "cache", "info"],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for line in result.stdout.split('\n')[:3]:  # First 3 lines
            if line.strip():
                print(f"   {line.strip()}")

    # Check for cached introdl
    result = subprocess.run([sys.executable, "-m", "pip", "cache", "list", "introdl"],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0 and result.stdout.strip():
        print(f"   ⚠️  Cached introdl found:")
        for line in result.stdout.split('\n')[:3]:
            if line.strip():
                print(f"     {line.strip()}")
    else:
        print(f"   ✓ No cached introdl")

except Exception as e:
    print(f"   ✗ Error checking cache: {e}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

# Summary and recommendations
print("\nSUMMARY:")
if len(found_installations) > 1:
    print("⚠️  MULTIPLE INSTALLATIONS DETECTED - This is likely the problem!")
    print("   Python may be importing from the wrong location.")
    print("\nRECOMMENDATION:")
    print("   1. Uninstall all versions: pip uninstall introdl -y")
    print("   2. Clear pip cache: pip cache remove introdl")
    print("   3. Reinstall from source: pip install /path/to/introdl --no-cache-dir")
elif len(found_installations) == 1:
    inst = found_installations[0]
    if inst['version'] != '1.5.16':
        print(f"⚠️  WRONG VERSION INSTALLED (found {inst['version']}, expected 1.5.16)")
        print("\nRECOMMENDATION:")
        print("   1. Force reinstall: pip install /path/to/introdl --force-reinstall --no-cache-dir")
else:
    print("✗ NO INSTALLATION FOUND")
    print("\nRECOMMENDATION:")
    print("   1. Install from source: pip install /path/to/introdl")