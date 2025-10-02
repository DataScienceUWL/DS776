#!/usr/bin/env python3
"""
OpenRouter API Key Distribution Script

This script automatically retrieves and configures an OpenRouter API key for students
in the DS776 Deep Learning course. It uses a hashed email mapping to ensure each
student receives a unique key without exposing the mapping publicly.

Usage:
    python distribute_openrouter_key.py

The script will:
    1. Detect the student's email from CoCalc environment
    2. Retrieve the assigned OpenRouter API key
    3. Update the api_keys.env file in home_workspace

Author: DS776 Course Tools
Date: 2025-10-02
"""

import os
import sys
import hashlib
import json
from pathlib import Path

# URL to the hosted key mapping (GitHub Pages)
KEY_MAPPING_URL = "https://datascienceuwl.github.io/ds776-keys/key_mapping.json"

def get_student_email():
    """
    Retrieve student email from CoCalc environment variables.

    CoCalc sets COCALC_EMAIL to the student's login email.
    """
    email = os.environ.get('COCALC_EMAIL', '').strip().lower()

    if not email:
        print("⚠️  Error: Could not detect student email from environment")
        print("   This script must be run in CoCalc")
        print("   Environment variable COCALC_EMAIL is not set")
        sys.exit(1)

    return email

def fetch_key_mapping():
    """
    Download the key mapping from GitHub Pages.

    Returns:
        dict: Mapping of email hashes to API keys
    """
    try:
        import urllib.request
        with urllib.request.urlopen(KEY_MAPPING_URL, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"⚠️  Error: Failed to download key mapping")
        print(f"   URL: {KEY_MAPPING_URL}")
        print(f"   Error: {e}")
        sys.exit(1)

def get_assigned_key(email, mapping):
    """
    Retrieve the API key assigned to this student.

    Args:
        email: Student email address
        mapping: Key mapping dictionary

    Returns:
        str: Assigned API key
    """
    email_hash = hashlib.sha256(email.encode()).hexdigest()
    key = mapping.get(email_hash)

    if not key:
        print(f"⚠️  Error: No API key found for email: {email}")
        print("   Please contact the instructor")
        sys.exit(1)

    return key

def update_env_file(api_key):
    """
    Update the api_keys.env file with the assigned OpenRouter API key.

    Args:
        api_key: The OpenRouter API key to write
    """
    env_file = Path.home() / "home_workspace" / "api_keys.env"

    if not env_file.exists():
        print(f"⚠️  Error: api_keys.env not found at: {env_file}")
        print("   Please ensure your home_workspace is properly configured")
        sys.exit(1)

    # Read current content
    with open(env_file, 'r') as f:
        lines = f.readlines()

    # Update OPENROUTER_API_KEY line
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith('OPENROUTER_API_KEY='):
            old_value = line.strip().split('=', 1)[1] if '=' in line else ''
            lines[i] = f'OPENROUTER_API_KEY={api_key}\n'
            updated = True
            break

    # Write back
    with open(env_file, 'w') as f:
        f.writelines(lines)

    return updated

def main():
    """
    Main distribution workflow.
    """
    print("=" * 60)
    print("DS776 OpenRouter API Key Distribution")
    print("=" * 60)

    # Step 1: Get student email
    print("\n1. Detecting student email...")
    email = get_student_email()
    print(f"   ✓ Email: {email}")

    # Step 2: Download key mapping
    print("\n2. Downloading key mapping...")
    mapping = fetch_key_mapping()
    print(f"   ✓ Retrieved mapping for {len(mapping)} students")

    # Step 3: Get assigned key
    print("\n3. Retrieving your assigned API key...")
    api_key = get_assigned_key(email, mapping)
    print(f"   ✓ Key retrieved: {api_key[:20]}...")

    # Step 4: Update env file
    print("\n4. Updating api_keys.env file...")
    updated = update_env_file(api_key)
    if updated:
        print("   ✓ OpenRouter API key updated successfully")
    else:
        print("   ⚠️  Warning: OPENROUTER_API_KEY line not found in api_keys.env")
        print("   You may need to manually add your key")

    # Success message
    print("\n" + "=" * 60)
    print("✓ SUCCESS")
    print("=" * 60)
    print("Your OpenRouter API key has been configured!")
    print(f"\nKey: {api_key}")
    print("\nThe key is now available in:")
    print("  ~/home_workspace/api_keys.env")
    print("\nYou can now use OpenRouter models in your notebooks.")
    print("=" * 60)

if __name__ == "__main__":
    main()
