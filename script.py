# script.py
import sys
from fchk_read import (
    load_basis_from_fchk,
    get_orbital_count_fchk,
    get_orbital_energies_and_occupations_fchk,
    load_cmos_from_fchk,
    compute_cube_data_fchk,
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py file.fchk")
        sys.exit(1)

    fchk_path = sys.argv[1]

    # 1) Basis and atoms
    basis, coords, atom_info = load_basis_from_fchk(fchk_path)
    print(f"Basis functions: {len(basis)}")
    print(f"Atoms: {len(atom_info)}")


if __name__ == "__main__":
    main()

