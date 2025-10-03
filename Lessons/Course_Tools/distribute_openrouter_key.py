#!/usr/bin/env python3
"""
OpenRouter API Key Distribution Script (Encrypted Version)

This script retrieves and decrypts an OpenRouter API key for students.
Each key is encrypted with its project ID, so only the correct student
can decrypt their assigned key.

Usage:
    python distribute_openrouter_key_encrypted.py

The script will:
    1. Detect the student's project ID from CoCalc environment
    2. Download the encrypted key mapping from GitHub Pages
    3. Decrypt the API key using the project ID
    4. Update the api_keys.env file
    5. Create a backup file

Author: DS776 Course Tools
Date: 2025-10-02
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from base64 import b64decode
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# URL to the hosted encrypted key mapping (GitHub Pages)
KEY_MAPPING_URL = "https://datascienceuwl.github.io/ds776-keys/encrypted_key_mapping.json"

def get_project_id():
    """
    Retrieve student's project ID from CoCalc environment variables.
    """
    project_id = os.environ.get('COCALC_PROJECT_ID', '').strip()

    if not project_id:
        print("⚠️  Error: Could not detect project ID from environment")
        print("   This script must be run in CoCalc")
        print("   Environment variable COCALC_PROJECT_ID is not set")
        sys.exit(1)

    return project_id

def fetch_encrypted_mapping():
    """
    Download the encrypted key mapping from GitHub Pages.

    Returns:
        dict: Mapping of project_id hashes to encrypted API key data
    """
    try:
        import urllib.request
        with urllib.request.urlopen(KEY_MAPPING_URL, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"⚠️  Error: Failed to download encrypted key mapping")
        print(f"   URL: {KEY_MAPPING_URL}")
        print(f"   Error: {e}")
        sys.exit(1)

def derive_key_from_project_id(project_id: str) -> bytes:
    """
    Derive a 256-bit encryption key from project ID.

    Must match the encryption method used in generate_encrypted_mapping.py
    """
    salt = b'ds776-openrouter-2025'

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256 bits
        salt=salt,
        iterations=100000,
    )

    return kdf.derive(project_id.encode())

def decrypt_api_key(encrypted_data: dict, project_id: str) -> str:
    """
    Decrypt an API key using the project ID.

    Args:
        encrypted_data: Dict with 'ciphertext' and 'nonce' (base64-encoded)
        project_id: Student's project ID (used to derive decryption key)

    Returns:
        Decrypted API key string
    """
    try:
        # Derive decryption key from project ID
        key = derive_key_from_project_id(project_id)

        # Create AESGCM cipher
        aesgcm = AESGCM(key)

        # Decode base64 values
        ciphertext = b64decode(encrypted_data['ciphertext'])
        nonce = b64decode(encrypted_data['nonce'])

        # Decrypt
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        return plaintext.decode('utf-8')

    except Exception as e:
        print(f"⚠️  Error: Failed to decrypt API key")
        print(f"   This may mean you're not assigned a key yet")
        print(f"   Error: {e}")
        sys.exit(1)

def get_assigned_key(project_id: str, encrypted_mapping: dict) -> str:
    """
    Retrieve and decrypt the API key assigned to this student's project.

    Args:
        project_id: Student's CoCalc project ID
        encrypted_mapping: Encrypted key mapping dictionary

    Returns:
        str: Decrypted API key
    """
    project_id_hash = hashlib.sha256(project_id.encode()).hexdigest()
    encrypted_data = encrypted_mapping.get(project_id_hash)

    if not encrypted_data:
        print(f"⚠️  Error: No API key found for project ID: {project_id}")
        print("   Please contact the instructor")
        sys.exit(1)

    # Decrypt the API key
    api_key = decrypt_api_key(encrypted_data, project_id)
    return api_key

def update_env_file(api_key):
    """
    Update the api_keys.env file with the assigned OpenRouter API key.
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
            lines[i] = f'OPENROUTER_API_KEY={api_key}\n'
            updated = True
            break

    # Write back
    with open(env_file, 'w') as f:
        f.writelines(lines)

    return updated

def create_backup_file(api_key):
    """
    Create a backup file with the API key for safekeeping.
    """
    backup_file = Path.home() / "home_workspace" / "OPENROUTER_API_KEY.txt"

    try:
        with open(backup_file, 'w') as f:
            f.write(api_key + '\n')
        return True
    except Exception as e:
        print(f"   ⚠️  Warning: Could not create backup file: {e}")
        return False

def main():
    """
    Main distribution workflow.
    """
    print("=" * 60)
    print("DS776 OpenRouter API Key Distribution (Encrypted)")
    print("=" * 60)

    # Step 1: Get project ID
    print("\n1. Detecting project ID...")
    project_id = get_project_id()
    print(f"   ✓ Project ID: {project_id}")

    # Step 2: Download encrypted mapping
    print("\n2. Downloading encrypted key mapping...")
    encrypted_mapping = fetch_encrypted_mapping()
    print(f"   ✓ Retrieved encrypted mappings for {len(encrypted_mapping)} students")

    # Step 3: Decrypt assigned key
    print("\n3. Decrypting your assigned API key...")
    api_key = get_assigned_key(project_id, encrypted_mapping)
    print(f"   ✓ Key decrypted: {api_key[:20]}...")

    # Step 4: Update env file
    print("\n4. Updating api_keys.env file...")
    updated = update_env_file(api_key)
    if updated:
        print("   ✓ OpenRouter API key updated successfully")
    else:
        print("   ⚠️  Warning: OPENROUTER_API_KEY line not found in api_keys.env")
        print("   You may need to manually add your key")

    # Step 5: Create backup file
    print("\n5. Creating backup file...")
    if create_backup_file(api_key):
        print("   ✓ Backup file created: ~/home_workspace/OPENROUTER_API_KEY.txt")

    # Success message
    print("\n" + "=" * 60)
    print("✓ SUCCESS")
    print("=" * 60)
    print("Your OpenRouter API key has been configured!")
    print(f"\nKey: {api_key}")
    print("\nThe key is now available in:")
    print("  ~/home_workspace/api_keys.env")
    print("  ~/home_workspace/OPENROUTER_API_KEY.txt (backup)")
    print("\nYou can now use OpenRouter models in your notebooks.")
    print("\nSecurity: Your key was encrypted and only your project")
    print("          ID could decrypt it. The mapping file is safe")
    print("          to host publicly.")
    print("=" * 60)

if __name__ == "__main__":
    main()
