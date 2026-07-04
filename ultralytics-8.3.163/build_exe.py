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
    print(f"[build_exe] Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)

    if result.returncode != 0:
        print(f"[build_exe] PyInstaller exited with code {result.returncode}")
        sys.exit(result.returncode)

    dist_dir = PROJECT_ROOT / "dist"
    exe_path = dist_dir / "行人检测系统.exe"
    if exe_path.exists():
        print(f"[build_exe] Build successful!")
        print(f"[build_exe] Output: {exe_path}")
        print()
        print("=" * 60)
        print(" 部署目录结构:")
        print("=" * 60)
        print(f"  {dist_dir}/")
        print(f"    ├── 行人检测系统.exe")
        print(f"    ├── models/          ← 存放 .pt 模型文件")
        print(f"    ├── config/          ← 存放 .yaml 配置文件")
        print(f"    └── videos/          ← 存放测试视频文件")
        print("=" * 60)
    else:
        print(f"[build_exe] Warning: EXE not found at expected location: {exe_path}")
        print(f"[build_exe] Please check 'dist/' directory for output.")


if __name__ == "__main__":
    main()
