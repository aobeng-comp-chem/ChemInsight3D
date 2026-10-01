# ChemInsight3D

ChemInsight3D is a Python program for visualizing molecular orbitals and electron density in 3D. It reads wavefunctions from Gaussian (`.fchk`/`.fck`), Molden (`.molden`) and NBO (`.47`/`.31` with a key file) output. From them it computes orbital and density grids (cube data), localizes orbitals with the Pipek–Mezey method, and displays the results. It also opens existing `.cube` files.

## Download

Ready-to-run bundles are attached to each [release](https://github.com/aobeng-comp-chem/ChemInsight3D/releases). They contain their own Python and every package, so nothing else needs to be installed.

**Windows (64-bit):** download `ChemInsight3D-windows.zip`, unzip it, and run `ChemInsight3D\ChemInsight3D.exe`.

**Linux x86-64 (including WSL):** download `ChemInsight3D-linux-x86_64.tar.gz` and extract it on Linux with `tar`:

```bash
tar xzf ChemInsight3D-linux-x86_64.tar.gz -C ~
~/ChemInsight3D/ChemInsight3D
```

Do not extract the Linux archive with a Windows tool (Explorer, 7-Zip, WinRAR), even when you plan to run it from WSL. From WSL, extract it onto the Linux file system (for example `~`) rather than `/mnt/c`; it also starts faster there.

## Code overview

| Module | Purpose |
|---|---|
| `ChemInsight3D.py` | Viewer application (PyQt5 + PyVista) and program entry point |
| `fchk_read.py`, `read_molden.py`, `nbo_read.py` | File readers. All three return the same basis, geometry and orbital data structures |
| `localization_io.py` | Pipek–Mezey localization, Fock matrices and occupations |
| `density_analysis.py` | Total, alpha, beta and spin density grids |
| `grid_utils.py` | Builds cube grids and evaluates orbitals on them |
| `overlap_matrix.py` | AO overlap matrix |
| `angular_funct.py`, `bas_dict.py` | Angular parts of spherical Gaussian basis functions |
| `source_cache.py` | Caches parsed file data until the file changes |
| `path_utils.py` | Converts between Windows and WSL paths |
| `rebuild_file47.py`, `check_basis_ordering.py` | Tools for NBO `.47` files |
| `*.cpp`, `build_native_extensions.py` | Optional C++ extensions and their build script |
| `tests/` | Test suite |

## Dependencies

The program is developed on Linux/WSL with Python 3.10.

- **Required:** `numpy`, `scipy`, `pandas`, `PyQt5`, `pyvista`, `pyvistaqt`, `vtk`, `matplotlib`, `pillow`
- **To build the C++ extensions:** `pybind11`, plus a C++17 compiler with OpenMP (for example `g++`)
- **To run the tests:** `pytest`

`requirements.txt` pins the exact version of every package, so later releases cannot change how the program behaves. Install into a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Building the C++ extensions (optional)

The C++ extensions speed up overlap integrals, grid evaluation and localization. Without them the program falls back to slower Python code.

```bash
sudo apt install build-essential   # Ubuntu: installs the compiler
python3 build_native_extensions.py
```

On Windows the script uses the MSVC compiler, which it finds automatically. Install the Visual Studio Build Tools with the C++ workload first:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
python build_native_extensions.py
```

Rebuild with the same Python interpreter you use to run the program. These environment variables change the build:

- `CXX`: the compiler to use.
- `NO_OPENMP=1`: build without OpenMP.
- `NO_MARCH_NATIVE=1`: do not optimize for the CPU of this machine.

To check that the extensions load, run `python3 ChemInsight3D.py --self-check`.

## Standalone bundle

`build_bundle.sh` uses PyInstaller to build a folder containing its own Python, every package at the version in `requirements.txt`, and portable builds of the C++ extensions. Nothing installed on the target machine is used, so package updates cannot break it.

```bash
./build_bundle.sh                               # uses python3 (must be Python 3.10)
PYTHON=~/anaconda3/bin/python ./build_bundle.sh # or choose the Python to build with
```

The result is `build_dist/ChemInsight3D/` (about 1.1 GB). Copy the whole folder to another machine and run:

```bash
build_dist/ChemInsight3D/ChemInsight3D               # open the viewer
build_dist/ChemInsight3D/ChemInsight3D a.cube        # open cube files at startup
build_dist/ChemInsight3D/ChemInsight3D --self-check  # check that the C++ extensions load
```

The bundle runs on Linux x86-64 (including WSL) with a graphical desktop. It contains symbolic links, so copy it with Linux tools (`cp -a`, `rsync -a`, or a `tar` archive), not through Windows.

**Windows bundle:** `build_bundle.ps1` does the same on Windows. It needs Python 3.12 from python.org and the Visual Studio Build Tools (see above). Run it from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File build_bundle.ps1        # build
powershell -ExecutionPolicy Bypass -File build_bundle.ps1 -Zip   # build and zip
```

The result is `%LOCALAPPDATA%\ChemInsight3D-build\dist\ChemInsight3D\ChemInsight3D.exe`, with `ChemInsight3D-windows.zip` beside it when `-Zip` is given. Copy the whole `ChemInsight3D` folder, or the zip, to another 64-bit Windows machine and run `ChemInsight3D.exe`. Use `-WorkDir` to build somewhere else.

## How to run

**Viewer:**

```bash
python3 ChemInsight3D.py                 # open an empty window
python3 ChemInsight3D.py a.cube b.cube   # open cube files at startup
```

To compute orbitals from a wavefunction file, load the `.fchk`, `.molden` or `.47`/`.31` file from inside the program.

For NBO input, keep the key files next to the basis file with the same base name, for example `mol.47` with `mol.33` and `mol.40`.

For Molden input, the file has to declare `[5D7F]` or `[9G]` if it uses spherical functions.

**Command-line tools:**

```bash
# Check orbitals and run localization on a source file
python3 localization_io.py mol.fchk
python3 localization_io.py mol.47 --key mol.40 --space occupied_valence
python3 localization_io.py mol.molden --space range --range 49-60 --use-native

# NBO .47 utilities
python3 rebuild_file47.py input.47 output.47
python3 check_basis_ordering.py mol.47
```

Run `python3 localization_io.py --help` for all options.

**Tests:**

```bash
python3 -m pytest -q tests
```

The Fe(CO)₅ regression tests in `tests/test_feco5_regression.py` read these files, and are skipped when the files are missing:

- `test_files/closed-shell/gaussian/feco5-hf.fchk`
- `test_files/closed-shell/cfour/MOLDEN_new.molden`
- `test_files/closed-shell/cfour/MOLDEN.47`
- `test_files/closed-shell/cfour/MOLDEN.40`

## Author

ChemInsight3D is written and maintained by **Augustine Obeng** ([@aobeng-comp-chem](https://github.com/aobeng-comp-chem)), Department of Chemistry, University at Buffalo, The State University of New York. ORCID: [0009-0005-6794-2259](https://orcid.org/0009-0005-6794-2259).

For questions, bug reports or suggestions, open an [issue](https://github.com/aobeng-comp-chem/ChemInsight3D/issues) or email aobeng370@gmail.com.

## License

Copyright (C) 2026 Augustine Obeng

ChemInsight3D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3 or (at your option) any later version. It is distributed in the hope that it will be useful, but without any warranty. See [LICENSE](LICENSE) for the full terms.
