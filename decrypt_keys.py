"""
Utility to decrypt keys and load into environment.
Used by the bot to read encrypted .env files.
"""
import os
import sys
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Dict

def generate_key_from_password(password: str, salt: bytes) -> bytes:
    """Generate encryption key from password"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def decrypt_value(encrypted_value: str, fernet: Fernet) -> str:
    """Decrypt a value"""
    return fernet.decrypt(encrypted_value.encode()).decode()

def load_encrypted_env(env_path: str, password: str) -> Dict[str, str]:
    """Load and decrypt .env file"""
    # Load salt
    salt_file = env_path.replace('.env', '.salt')
    if not os.path.exists(salt_file):
        raise FileNotFoundError(f'Salt file not found: {salt_file}')

    with open(salt_file, 'rb') as f:
        salt = f.read()

    # Generate decryption key
    key = generate_key_from_password(password, salt)
    fernet = Fernet(key)

    # Read .env file
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key_name, value = line.split('=', 1)
                key_name = key_name.strip()
                value = value.strip()

                # Check if encrypted (starts with ENC:)
                if value.startswith('ENC:'):
                    encrypted = value[4:]  # Remove 'ENC:' prefix
                    try:
                        decrypted = decrypt_value(encrypted, fernet)
                        env_vars[key_name] = decrypted
                    except Exception as e:
                        hint = (
                            "Usually: wrong master password, or grok.salt does not match this grok.env "
                            "(e.g. re-encrypted once with a new salt). "
                            f"({type(e).__name__})"
                        )
                        msg = str(e).strip() or hint
                        raise ValueError(f'Failed to decrypt {key_name}: {msg}') from e
                else:
                    env_vars[key_name] = value

    return env_vars

def load_env_with_decryption(env_path: str = '.env', password: str = None) -> Dict[str, str]:
    """
    Load .env file with automatic decryption support.
    If password is None, will use MASTER_PASSWORD env var or prompt.
    """
    if not os.path.exists(env_path):
        return {}

    # Check if file has encrypted values
    with open(env_path, 'r') as f:
        content = f.read()

    has_encrypted = 'ENC:' in content

    if not has_encrypted:
        # No encryption, load normally
        env_vars = {}
        for line in content.split('\n'):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
        return env_vars

    # Has encrypted values, need password
    if password is None:
        password = os.getenv('MASTER_PASSWORD')
        if isinstance(password, str):
            password = password.strip() or None  # avoid copy/paste trailing newline breaking decrypt

    if password is None:
        if not sys.stdin.isatty():
            print(
                "[INFO] grok.env has encrypted values but no MASTER_PASSWORD (non-interactive). "
                "Skipping decryption; use plaintext keys or set MASTER_PASSWORD to trade.",
                file=sys.stderr,
            )
            return {}
        import getpass

        password = getpass.getpass("Enter master password to decrypt keys: ")

    return load_encrypted_env(env_path, password)

if __name__ == '__main__':
    # Test decryption
    import getpass
    password = getpass.getpass('Enter master password: ')

    try:
        env_vars = load_encrypted_env('.env', password)
        print('\n✅ Successfully decrypted .env file!')
        print(f'Loaded {len(env_vars)} variables:')
        for key in env_vars:
            if 'KEY' in key.upper() or 'SECRET' in key.upper():
                print(f'  {key}: {"*" * 20}')
            else:
                print(f'  {key}: {env_vars[key]}')
    except Exception as e:
        print(f'\n❌ Decryption failed: {e}')
