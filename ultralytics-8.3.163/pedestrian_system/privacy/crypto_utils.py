from __future__ import annotations

from pathlib import Path

try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None


class CryptoManager:
    def __init__(self, key_path: str = "secret.key"):
        if Fernet is None:
            raise RuntimeError("cryptography is not installed. Please install it to enable encryption.")

        self.key_path = Path(key_path)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key = self._load_or_create_key()
        self.fernet = Fernet(self.key)

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            key = self.key_path.read_bytes().strip()
            if key:
                return key

        key = Fernet.generate_key()
        self.key_path.write_bytes(key)
        return key

    def encrypt(self, data: bytes) -> bytes:
        return self.fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        return self.fernet.decrypt(data)
