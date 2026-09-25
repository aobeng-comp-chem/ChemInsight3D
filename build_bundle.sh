#!/usr/bin/env bash
# Build a standalone ChemInsight3D bundle with PyInstaller.
#
# The bundle contains its own Python, every package at the version pinned in
# requirements.txt, and the C++ extensions, so nothing installed on the target
# machine can change or break it. It runs on Linux x86-64 only.
#
# Usage:
#   ./build_bundle.sh                       # uses python3
#   PYTHON=~/anaconda3/bin/python ./build_bundle.sh
#
# Output: build_dist/chemview/chemview  (run it directly, or pass .cube files)
set -euo pipefail

cd "$(dirname "$0")"
ROOT=$(pwd)
PYTHON=${PYTHON:-python3}
BUILD_TOOLS=(pyinstaller==6.22.3 pyinstaller-hooks-contrib==2026.7 altgraph==0.17.5)

BUILD=$ROOT/build
VENV=$ROOT/.build-venv
STAGE=$BUILD/stage
DIST=$ROOT/build_dist

export PYTHONNOUSERSITE=1   # ignore packages in ~/.local

echo "==> Build environment ($("$PYTHON" --version))"
if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r requirements.txt "${BUILD_TOOLS[@]}"

echo "==> Staging sources"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp ./*.py ./*.cpp "$STAGE/"

echo "==> Compiling portable C++ extensions"
# Portable build (no -march=native) so the bundle runs on other CPUs.
# Built in the staging folder so the local .so files are left untouched.
(cd "$STAGE" && NO_MARCH_NATIVE=1 "$VENV/bin/python" build_native_extensions.py)

EXT_SUFFIX=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
ADD_BINARIES=()
for ext in overlap_matrix electron_density_opt_omp localization_native; do
    ADD_BINARIES+=(--add-binary "$STAGE/$ext$EXT_SUFFIX:.")
done

echo "==> Running PyInstaller"
rm -rf "$DIST/chemview"
"$VENV/bin/python" -m PyInstaller \
    --noconfirm \
    --onedir \
    --name chemview \
    --distpath "$DIST" \
    --workpath "$BUILD/pyinstaller" \
    --specpath "$BUILD" \
    --paths "$STAGE" \
    "${ADD_BINARIES[@]}" \
    --hidden-import electron_density_opt_omp \
    --hidden-import localization_native \
    --exclude-module tkinter \
    --exclude-module pytest \
    "$STAGE/chemview.py"

echo "==> Checking the C++ extensions load inside the bundle"
"$DIST/chemview/chemview" --self-check

echo
echo "==> Done: $DIST/chemview/chemview ($(du -sh "$DIST/chemview" | cut -f1))"
