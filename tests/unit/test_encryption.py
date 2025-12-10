"""Unit tests for encryption utilities."""

import os
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet, InvalidToken

from ragitect.utils.encryption import decrypt_value, encrypt_value, get_encryption_key


class TestEncryption:
    """Test suite for encryption utilities."""

    def test_get_encryption_key_success(self):
        """Test successful retrieval of encryption key."""
        test_key = Fernet.generate_key().decode()
        with patch.dict(os.environ, {"ENCRYPTION_KEY": test_key}):
            key = get_encryption_key()
            assert key == test_key.encode()

    def test_get_encryption_key_missing(self):
        """Test error when ENCRYPTION_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="ENCRYPTION_KEY environment variable not set"
            ):
                get_encryption_key()

    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption and decryption work correctly."""
        test_key = Fernet.generate_key().decode()
        plaintext = "sk-test-api-key-12345"

        with patch.dict(os.environ, {"ENCRYPTION_KEY": test_key}):
            encrypted = encrypt_value(plaintext)
            assert encrypted != plaintext
            assert len(encrypted) > 0

            decrypted = decrypt_value(encrypted)
            assert decrypted == plaintext

    def test_encrypt_empty_string(self):
        """Test encrypting empty string returns empty string."""
        test_key = Fernet.generate_key().decode()

        with patch.dict(os.environ, {"ENCRYPTION_KEY": test_key}):
            encrypted = encrypt_value("")
            assert encrypted == ""

    def test_decrypt_empty_string(self):
        """Test decrypting empty string returns empty string."""
        test_key = Fernet.generate_key().decode()

        with patch.dict(os.environ, {"ENCRYPTION_KEY": test_key}):
            decrypted = decrypt_value("")
            assert decrypted == ""

    def test_decrypt_invalid_ciphertext(self):
        """Test that decrypting invalid ciphertext raises error."""
        test_key = Fernet.generate_key().decode()

        with patch.dict(os.environ, {"ENCRYPTION_KEY": test_key}):
            with pytest.raises(InvalidToken):
                decrypt_value("invalid-encrypted-text")

    def test_decrypt_with_wrong_key(self):
        """Test that decrypting with wrong key fails."""
        key1 = Fernet.generate_key().decode()
        key2 = Fernet.generate_key().decode()
        plaintext = "test-secret"

        # Encrypt with key1
        with patch.dict(os.environ, {"ENCRYPTION_KEY": key1}):
            encrypted = encrypt_value(plaintext)

        # Try to decrypt with key2
        with patch.dict(os.environ, {"ENCRYPTION_KEY": key2}):
            with pytest.raises(InvalidToken):
                decrypt_value(encrypted)
