#!/usr/bin/env python3
"""
Build the native pybind11 extensions for the current Python interpreter.

Usage:
    python3 build_native_extensions.py

Optional environment variables:
    CXX=clang++
    PYBIND11_INCLUDE=/path/to/pybind11/include-parent
    NO_OPENMP=1
    NO_MARCH_NATIVE=1  # disable -march=native (e.g. building for another host)
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parent
CXX = os.environ.get("CXX", "g++")
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _unique_paths(paths):
    unique = []
    seen = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _python_include_dirs():
    paths = sysconfig.get_paths()
    return _unique_paths(
        p for key in ("include", "platinclude")
        if (p := paths.get(key))
    )


def _pybind11_include_dirs():
    candidates = []

    if env_path := os.environ.get("PYBIND11_INCLUDE"):
        candidates.append(Path(env_path))

    try:
        import pybind11  # type: ignore
    except Exception:
        pybind11 = None

    if pybind11 is not None:
        candidates.append(Path(pybind11.get_include()))

    for base in (Path("/usr/include"), Path("/usr/local/include")):
        if (base / "pybind11").exists():
            candidates.append(base)

    return _unique_paths(candidates)


def _common_compile_flags():
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC", "-DNDEBUG"]
    if not os.environ.get("NO_OPENMP") and sys.platform.startswith("linux"):
        flags.append("-fopenmp")
    if not os.environ.get("NO_MARCH_NATIVE"):
        flags.append("-march=native")
    return flags


def _compile(source_name: str):
    source = ROOT / source_name
    output = ROOT / f"{source.stem}{EXT_SUFFIX}"

    pybind11_includes = _pybind11_include_dirs()
    if not pybind11_includes:
        raise SystemExit(
            "pybind11 headers were not found.\n"
            "Install them first with one of:\n"
            "  python3 -m pip install pybind11\n"
            "  sudo apt install pybind11-dev\n"
            "Then rerun:\n"
            "  python3 build_native_extensions.py"
        )

    include_flags = []
    for inc in [*_python_include_dirs(), *pybind11_includes]:
        include_flags.extend(["-I", str(inc)])

    cmd = [
        CXX,
        *(_common_compile_flags()),
        *include_flags,
        str(source),
        "-o",
        str(output),
    ]

    print(f"Building {source.name} -> {output.name}")
    print(" ", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    return output


def main():
    built = []
    for source_name in (
        "overlap_matrix.cpp",
        "electron_density_opt_omp.cpp",
        "localization_native.cpp",
    ):
        built.append(_compile(source_name))

    print("\nBuild complete:")
    for output in built:
        print(f"  {output.name}")


if __name__ == "__main__":
    main()
