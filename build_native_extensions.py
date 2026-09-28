#!/usr/bin/env python3
"""
Build the native pybind11 extensions for the current Python interpreter.

Usage:
    python3 build_native_extensions.py

On Linux this uses g++ (or $CXX). On Windows it uses the MSVC compiler
(cl.exe) from Visual Studio or the Visual Studio Build Tools, found
automatically, and copies the OpenMP runtime DLL next to the extensions.

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
import shutil
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parent
IS_WINDOWS = sys.platform == "win32"
CXX = os.environ.get("CXX", "cl" if IS_WINDOWS else "g++")
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if IS_WINDOWS else ".so")
OPENMP_DLL = "vcomp140.dll"   # MSVC OpenMP runtime, redistributable


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


def _find_msvc_install():
    """Visual Studio installation folder that has the C++ tools, via vswhere."""
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    install = ""
    if vswhere.exists():
        install = subprocess.run(
            [str(vswhere), "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    if not install:
        raise SystemExit(
            "The MSVC C++ compiler was not found.\n"
            "Install the Visual Studio Build Tools with the C++ workload, e.g.:\n"
            "  winget install Microsoft.VisualStudio.2022.BuildTools --override "
            "\"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools "
            "--includeRecommended\""
        )
    return Path(install)


def _msvc_environment():
    """
    Environment for running cl.exe. Uses the current one when cl is already on
    PATH (a Developer Command Prompt), otherwise loads vcvars64.bat.
    """
    if shutil.which("cl"):
        return None
    vcvars = _find_msvc_install() / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    output = subprocess.run(
        f'"{vcvars}" >nul && set', shell=True,
        capture_output=True, text=True, check=True,
    ).stdout
    env = {}
    for line in output.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            env[key] = value
    return env


def _msvc_command(source, output, include_dirs):
    flags = ["/nologo", "/O2", "/std:c++17", "/EHsc", "/bigobj", "/utf-8", "/MD", "/LD",
             "/DNDEBUG", "/D_USE_MATH_DEFINES"]
    if not os.environ.get("NO_OPENMP"):
        # The code uses `omp simd`, which plain /openmp (OpenMP 2.0) rejects.
        # /openmp:llvm would also work, but its runtime DLL is not
        # redistributable; /openmp:experimental uses vcomp140.dll, which is.
        flags.append("/openmp:experimental")
    python_libs = Path(sys.base_prefix) / "libs"
    return [
        CXX, *flags,
        *(f"/I{inc}" for inc in include_dirs),
        str(source),
        f"/Fe:{output}",
        f"/Fo:{output.with_suffix('.obj')}",
        "/link", f"/LIBPATH:{python_libs}",
    ]


def _copy_openmp_runtime():
    """Copy the OpenMP runtime DLL next to the built extensions."""
    if os.environ.get("NO_OPENMP"):
        return None
    redist = _find_msvc_install() / "VC" / "Redist" / "MSVC"
    matches = sorted(redist.glob(f"*/x64/*.OpenMP/{OPENMP_DLL}"))
    if not matches:
        print(f"Warning: {OPENMP_DLL} not found under {redist}; the extensions "
              "will only load where that DLL is on PATH.")
        return None
    target = ROOT / OPENMP_DLL
    shutil.copy2(matches[-1], target)
    return target


def _compile(source_name: str, msvc_env=None):
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

    include_dirs = [*_python_include_dirs(), *pybind11_includes]

    if IS_WINDOWS:
        cmd = _msvc_command(source, output, include_dirs)
        if msvc_env:
            # Windows looks the program up on this process's PATH, not env's.
            search_path = msvc_env.get("Path") or msvc_env.get("PATH")
            cmd[0] = shutil.which(CXX, path=search_path) or CXX
    else:
        include_flags = []
        for inc in include_dirs:
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
    subprocess.run(cmd, check=True, cwd=ROOT, env=msvc_env)
    return output


def main():
    msvc_env = _msvc_environment() if IS_WINDOWS else None
    built = []
    for source_name in (
        "overlap_matrix.cpp",
        "electron_density_opt_omp.cpp",
        "localization_native.cpp",
    ):
        built.append(_compile(source_name, msvc_env))

    if IS_WINDOWS and (runtime := _copy_openmp_runtime()):
        built.append(runtime)

    print("\nBuild complete:")
    for output in built:
        print(f"  {output.name}")


if __name__ == "__main__":
    main()
