"""Build the optional pybind11 C++ backend into the source package."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Configure CMake and place the compiled extension beside the Python package."""

    parser = argparse.ArgumentParser(
        description="Configure and build the optional in-place C++ backend."
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/native")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "src/disparity_deblur",
    )
    args = parser.parse_args()

    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError(
            "cmake is unavailable; run `uv sync --group native` first"
        )
    pybind11_dir = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--cmakedir"],
        text=True,
    ).strip()
    configure = [
        cmake,
        "-S",
        str(ROOT / "native"),
        "-B",
        str(args.build_dir),
        f"-Dpybind11_DIR={pybind11_dir}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={args.output_dir}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={args.output_dir}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={args.output_dir}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={args.output_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if shutil.which("ninja") is not None:
        configure.extend(("-G", "Ninja"))
    subprocess.run(configure, check=True)
    subprocess.run(
        [cmake, "--build", str(args.build_dir), "--config", "Release"],
        check=True,
    )
    extensions = sorted(args.output_dir.glob("_native*"))
    if not extensions:
        extensions = sorted(args.build_dir.rglob("_native*"))
    if not extensions:
        raise RuntimeError("native build completed without producing an extension")
    print(extensions[-1])


if __name__ == "__main__":
    main()
