"""
check_basis_ordering.py
========================
Reads an NBO basis file (.47/.31) and reports whether every atom's basis
functions are listed as one contiguous run (grouped by CENTER), or whether
they're fragmented -- e.g. extra shells for an atom appearing later in the
file, after other atoms' shells, as some Molcas-generated NBO files do.

Several parts of chem-insight3d (population analysis, Pipek-Mezey
localization) need to look up "which basis-function indices belong to atom
A" cheaply as a single [lo, hi] range. When a file isn't center-contiguous,
that lookup silently mis-partitions atoms unless it's specifically guarded
against -- this script lets you check a file ahead of time.

Usage:
    python check_basis_ordering.py FILE.47
"""

import sys

import nbo_read
from localization_io import analyze_center_contiguity


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_basis_ordering.py FILE.47")
        sys.exit(1)

    path = sys.argv[1]
    # parse_file47 (not load_basis_headless) -- this only needs each basis
    # function's raw CENTER field, so there's no reason to run it through
    # load_basis_headless's normalize_by_self_overlap/iterative refinement
    # pipeline (which rewrites the coefficients and is meant for actual
    # QC calculations, not an ordering check).
    final_basis, _, _, _ = nbo_read.parse_file47(path)
    report = analyze_center_contiguity(final_basis)

    print(f"File             : {path}")
    print(f"Basis functions  : {report['n_basis_functions']}")
    print(f"Atoms (distinct CENTER values) : {report['n_atoms']}")
    print()

    if report["contiguous"]:
        print("OK -- every atom's basis functions form one contiguous run.")
        return

    print(f"WARNING -- {len(report['fragmented_atoms'])} atom(s) have basis "
          f"functions split across more than one run:\n")

    for center in report["fragmented_atoms"]:
        runs = report["atoms"][center]
        print(f"  CENTER {center}: {len(runs)} runs")
        for lo, hi in runs:
            # Report 1-based "N" indices, matching the app's Basis
            # Functions table.
            print(f"      N {lo + 1}-{hi + 1}  ({hi - lo + 1} functions)")
        print()


if __name__ == "__main__":
    main()
