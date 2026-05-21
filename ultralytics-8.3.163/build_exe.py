from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_FILE = PROJECT_ROOT / "pedestrian_system.spec"


def main() -> None:
    if not SPEC_FILE.exists():
        raise SystemExit(f"Spec file not found: {SPEC_FILE}")

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--log-level=WARN",
        str(SPEC_FILE),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()