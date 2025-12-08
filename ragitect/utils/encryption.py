"""Encryption utilities for secure storage of API keys.

This module provides Fernet symmetric encryption for encrypting and decrypting
sensitive data like API keys before storing them in the database.
"""

import os

from cryptography.fernet import Fernet


def get_encryption_key() -> bytes:
    """Get encryption key from environment variable.

    Returns:
        bytes: Encryption key

    Raises:
        ValueError: If ENCRYPTION_KEY is not set in environment
    """
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        raise ValueError(
            "ENCRYPTION_KEY environment variable not set. "
            'Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        )
    return key.encode()


def encrypt_value(plaintext: str) -> str:
    """Encrypt a plaintext string.

    Args:
        plaintext: String to encrypt

    Returns:
        str: Encrypted string (base64 encoded)
    """
    if not plaintext:
        return plaintext

    key = get_encryption_key()
    f = Fernet(key)
    encrypted_bytes = f.encrypt(plaintext.encode())
    return encrypted_bytes.decode()


def decrypt_value(ciphertext: str) -> str:
    """Decrypt an encrypted string.

    Args:
        ciphertext: Encrypted string (base64 encoded)

    Returns:
        str: Decrypted plaintext string

    Raises:
        cryptography.fernet.InvalidToken: If decryption fails
    """
    if not ciphertext:
        return ciphertext

    key = get_encryption_key()
    f = Fernet(key)
    decrypted_bytes = f.decrypt(ciphertext.encode())
    return decrypted_bytes.decode()
