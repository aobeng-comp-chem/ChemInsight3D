"""
rebuild_file47.py
==================
Rewrite an NBO .47 basis file so every atom's basis functions form one
contiguous, CENTER-ordered block, instead of being interleaved with other
atoms' shells (as some Molcas-generated NBO files are -- run
check_basis_ordering.py first to see whether a given file needs this).

What gets rewritten
--------------------
$BASIS and $CONTRACT are regenerated from scratch in NBO's "ungrouped"
form -- one primitive-set entry per basis function, NCOMP=1 everywhere.
That's a standard, fully valid .47 representation (nbo_read.py's own
parser already recognizes and handles it via its is_ungrouped() check),
and sidesteps having to reconstruct the original d/f/g shell groupings.
$OVERLAP / $DENSITY / $FOCK are permuted (both rows and columns) to match
the new AO ordering. $GENNBO / $NBO / $COORD are copied through
byte-for-byte, since none of them depend on AO order.

What does NOT get rewritten
-----------------------------
Some .47 files carry additional AO-basis sections this codebase never
reads or checks against a working reference -- $LCAOMO, $KINETIC,
$NUCLEAR, $DIPOLE, or others. Copying them through unchanged would leave
them silently referencing the *old* AO ordering while $BASIS/$CONTRACT
now describe a different one -- worse than not having them at all. Any
such section is detected and dropped from the output, with a warning
naming it, rather than guessed at.

Usage:
    python rebuild_file47.py INPUT.47 OUTPUT.47
"""

import re
import sys

import numpy as np

import nbo_read
from localization_io import analyze_center_contiguity


# NBO's LABEL codes group by hundreds: 1/51 = s, 100s = p, 200s = d, ...
_LABEL_RANGES = [
    ("S", {1, 51}),
    ("P", range(100, 200)),
    ("D", range(200, 300)),
    ("F", range(300, 400)),
    ("G", range(400, 500)),
    ("H", range(500, 600)),
    # I (600s) and J (700s) shells aren't supported yet -- _angmom_letter
    # raises for them below rather than silently mis-writing $CONTRACT.
]
_COEFF_ARRAY_NAMES = {"S": "CS", "P": "CP", "D": "CD", "F": "CF",
                      "G": "CG", "H": "CH"}


def _angmom_letter(label):
    for letter, members in _LABEL_RANGES:
        if label in members:
            return letter
    raise ValueError(f"Unrecognized LABEL value: {label}")


def _fortran_float(v, frac_digits=12):
    """
    Format a float the way real .47 files do: Fortran's default E-editing,
    i.e. a normalized mantissa with a leading "0." (not a leading nonzero
    digit like Python's own scientific notation), frac_digits fractional
    digits, then 'E' and a signed 2-digit exponent -- e.g.
    "0.100000000000E+01", "-0.431408307543E-31".

    Verified against real NBO output: unpacking a genuine $OVERLAP block
    written this way reproduces an independently computed overlap matrix
    to ~1e-7; the naive "1.234...E+02"-style mantissa does not round-trip
    through nbo7 at all (this is what caused "SINP: error reading
    $OVERLAP" when the writer used Python's default float formatting).
    """
    if v == 0:
        return "0." + "0" * frac_digits + "E+00"
    sign = "-" if v < 0 else ""
    mantissa, exp_str = f"{abs(v):.{frac_digits - 1}E}".split("E")
    lead_digit, frac_part = mantissa.split(".")
    digits = (lead_digit + frac_part)[:frac_digits]
    exp = int(exp_str) + 1
    return f"{sign}0.{digits}E{exp:+03d}"


def _int_field(v, width=6):
    return f"{v:{width}d}"


def _float_field(v, width=20):
    return f"{_fortran_float(v):>{width}s}"


def _format_named_array(name, values, field_fn, per_line):
    """
    NBO's own convention for a "NAME = v1 v2 ..." declaration: the name is
    right-justified in an 8-char field followed by " =" (10 chars total,
    identical whether the array holds ints or floats), continuation lines
    are indented by exactly that same 10 spaces, and each value is
    right-justified in its own fixed-width field with NO separator between
    them -- verified by measuring exact character columns in real .47
    files (integer fields always end 6 columns apart; float fields always
    end 20 columns apart, regardless of a value's sign).
    """
    prefix = f"{name:>8s} ="
    indent = " " * 10
    chunks = [values[i:i + per_line] for i in range(0, len(values), per_line)]
    lines = [
        (prefix if k == 0 else indent) + "".join(field_fn(v) for v in chunk)
        for k, chunk in enumerate(chunks)
    ]
    return "\n".join(lines) + "\n"


def _format_int_array(name, values, per_line=11):
    return _format_named_array(name, values, _int_field, per_line)


def _format_float_array(name, values, per_line=3):
    return _format_named_array(name, values, _float_field, per_line)


def _format_bare_float_block(values, per_line=4):
    # $OVERLAP/$DENSITY/$FOCK have no "NAME =" declaration -- just rows of
    # right-justified 20-char fields, no separator, starting at column 0.
    chunks = [values[i:i + per_line] for i in range(0, len(values), per_line)]
    lines = ["".join(_float_field(v) for v in chunk) for chunk in chunks]
    return "\n".join(lines) + "\n"


def _extract_verbatim_block(content, start_keyword):
    """Return the exact text from start_keyword through its first following $END."""
    start = content.index(start_keyword)
    end = content.index("$END", start) + len("$END")
    return content[start:end]


def _find_unhandled_sections(content, handled):
    sections = re.findall(r'^\s*\$(\w+)', content, flags=re.MULTILINE)
    return sorted({s for s in sections if s.upper() not in handled and s.upper() != "END"})


def rebuild_file47(input_path, output_path):
    with open(input_path, "r") as f:
        content = f.read()

    header_lines = content.splitlines()
    header_line = header_lines[0]
    has_upper = "UPPER" in header_line.upper()

    basis_info, _, _, _ = nbo_read.parse_file47(input_path)
    nbas = len(basis_info)

    report = analyze_center_contiguity(basis_info)
    if report["contiguous"]:
        print(f"{input_path} is already center-contiguous -- copying it through unchanged.")
        with open(output_path, "w") as f:
            f.write(content)
        return

    order = sorted(range(nbas), key=lambda i: basis_info[i]["CENTER"])
    perm = np.array(order)
    reordered = [basis_info[i] for i in order]

    for bf in reordered:
        if len(bf["exps"]) != len(bf["coeffs"]):
            raise ValueError(
                f"Basis function N={bf['N']} has {len(bf['exps'])} exponents "
                f"but {len(bf['coeffs'])} contraction coefficients -- refusing "
                "to guess how to regenerate $CONTRACT for this file."
            )

    handled_keywords = {"GENNBO", "NBO", "COORD", "BASIS", "CONTRACT",
                        "OVERLAP", "DENSITY", "FOCK"}
    unhandled = _find_unhandled_sections(content, handled_keywords)
    if unhandled:
        print(
            f"WARNING: {input_path} contains section(s) not understood by this "
            f"tool -- {', '.join(unhandled)} -- these are AO-order-dependent but "
            "nbo_read.py never parses them, so their exact permutation convention "
            "can't be checked here. They are DROPPED from the output rather than "
            "silently left inconsistent with the new AO ordering."
        )

    # ---- $BASIS ----
    center_arr = [bf["CENTER"] for bf in reordered]
    label_arr = [bf["LABEL"] for bf in reordered]
    basis_block = (
        " $BASIS\n"
        + _format_int_array("CENTER", center_arr)
        + _format_int_array("LABEL", label_arr)
        + " $END\n"
    )

    # ---- $CONTRACT (ungrouped: one independent primitive set per AO) ----
    nprim_arr = [len(bf["exps"]) for bf in reordered]

    # Real .47 files reuse the same primitive block across multiple shell
    # entries whenever the exponents/coefficients are identical (e.g. the
    # px/py/pz components of one contraction, or repeated angular components
    # of a d/f/g/h shell all point at the same NPTR) -- NCOMP stays 1 per
    # entry, but NPTR duplicates. Skipping that dedup would give every AO its
    # own private copy of the primitives, inflating NEXP several-fold (e.g.
    # 3x for p, 5x for d, 7x for f, ...) versus the original file for no
    # numerical benefit, since the values are identical either way.
    exp_pool = []
    ptr_cache = {}
    nptr_arr = []
    next_ptr = 1
    for bf in reordered:
        letter = _angmom_letter(bf["LABEL"])
        key = (letter, tuple(bf["exps"]), tuple(bf["coeffs"]))
        ptr = ptr_cache.get(key)
        if ptr is None:
            ptr = next_ptr
            ptr_cache[key] = ptr
            exp_pool.extend(bf["exps"])
            next_ptr += len(bf["exps"])
        nptr_arr.append(ptr)
    nexp_total = next_ptr - 1

    coeff_pools = {letter: [0.0] * nexp_total for letter in _COEFF_ARRAY_NAMES}
    for (letter, exps, coeffs), ptr in ptr_cache.items():
        coeff_pools[letter][ptr - 1:ptr - 1 + len(exps)] = coeffs

    contract_block = " $CONTRACT\n" + _format_int_array("NSHELL", [nbas])
    contract_block += _format_int_array("NEXP", [nexp_total])
    contract_block += _format_int_array("NCOMP", [1] * nbas)
    contract_block += _format_int_array("NPRIM", nprim_arr)
    contract_block += _format_int_array("NPTR", nptr_arr)
    contract_block += _format_float_array("EXP", exp_pool)
    for letter in ["S", "P", "D", "F", "G", "H"]:
        contract_block += _format_float_array(_COEFF_ARRAY_NAMES[letter], coeff_pools[letter])
    contract_block += " $END\n"

    # ---- $OVERLAP / $DENSITY / $FOCK: permute rows+cols to the new AO order ----
    def packed_values(matrix):
        m = matrix[np.ix_(perm, perm)]
        if not has_upper:
            return m.reshape(-1)
        # UPPER means the file stores the upper triangle packed *column by
        # column* (Fortran convention: outer loop over columns j, inner loop
        # over rows i=1..j). For a symmetric matrix that produces the exact
        # same flat sequence as the lower triangle packed *row by row* in
        # NumPy's row-major layout -- which is what np.tril_indices gives.
        # Verified against ground truth: unpacking a real UPPER-flagged
        # $OVERLAP block via tril_indices matches an independently computed
        # overlap matrix to ~1e-7; unpacking the same data via triu_indices
        # does not (this codebase's own create_symmetric_matrix_vectorized
        # reader also assumes tril_indices, so this keeps read/write
        # consistent).
        return m[np.tril_indices(nbas)]

    is_open_shell_data, keyword_dict = nbo_read.process_47_file(input_path, nbas)
    matrix_blocks = ""

    if "$OVERLAP" in content:
        vals = packed_values(keyword_dict["OVERLAP"])
        matrix_blocks += " $OVERLAP\n" + _format_bare_float_block(list(vals)) + " $END\n"

    for base_name in ["DENSITY", "FOCK"]:
        if f"${base_name}" not in content:
            continue
        if is_open_shell_data:
            alpha_vals = packed_values(keyword_dict[f"{base_name}_ALPHA"])
            beta_vals = packed_values(keyword_dict[f"{base_name}_BETA"])
            vals = np.concatenate([alpha_vals, beta_vals])
        else:
            vals = packed_values(keyword_dict[base_name])
        matrix_blocks += f" ${base_name}\n" + _format_bare_float_block(list(vals)) + " $END\n"

    # ---- Assemble: original header/coord verbatim + regenerated blocks ----
    coord_block = _extract_verbatim_block(content, "$COORD")
    out = (
        header_lines[0] + "\n"
        + header_lines[1] + "\n"
        + " " + coord_block + "\n"
        + basis_block
        + contract_block
        + matrix_blocks
    )

    with open(output_path, "w") as f:
        f.write(out)

    print(f"Wrote {output_path}: {nbas} basis functions, "
          f"{report['n_atoms']} atoms, now center-contiguous.")


def main():
    if len(sys.argv) != 3:
        print("Usage: python rebuild_file47.py INPUT.47 OUTPUT.47")
        sys.exit(1)
    rebuild_file47(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
