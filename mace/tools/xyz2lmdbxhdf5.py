from ase.io.extxyz import read_extxyz
import io
import h5py
from tqdm import tqdm
from ase.db.core import connect
import numpy as np

def split_concatenated_xyz(filename):
    """
    Split XYZ file where structures are concatenated without blank lines
    Assumes standard XYZ format: atom_count, comment_line, then coordinates
    """
    structures = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    structure_count = 0

    while i < len(lines):
        line = lines[i].strip()

        # Check if this line is an atom count (should be a number)
        try:
            n_atoms = int(line)

            # Extract this structure: atom count + comment + atom coordinates
            if i + 1 + n_atoms < len(lines):
                structure_lines = lines[i:i+2+n_atoms]
                structure_text = ''.join(structure_lines)

                # Parse with ASE
                try:
                    structure_io = io.StringIO(structure_text)
                    atoms = next(read_extxyz(structure_io))
                    structures.append(atoms)
                    structure_count += 1
                    print(f"Structure {structure_count}: {n_atoms} atoms, energy={atoms.info.get('dft_energy', 'N/A')}")

                except Exception as e:
                    print(f"Failed to parse structure {structure_count + 1}: {e}")

                # Move to next structure
                i += 2 + n_atoms
            else:
                break

        except ValueError:
            # This line is not an atom count, skip it
            i += 1

    return structures

# Use it with your original code:
structures = split_concatenated_xyz('training_set.xyz')
file_name = 'test'
print(f"\nFound {len(structures)} structures")

batch_size = 50
if len(structures) > 0:
    with h5py.File(str(file_name) + '.h5', 'w') as z:
        z.attrs['drop_last'] = False

        for batch_idx in range(0, len(structures), batch_size):
            batch_structures = structures[batch_idx:batch_idx + batch_size]
            batch_group_name = f"config_batch_{batch_idx // batch_size}"
            batch_group = z.create_group(batch_group_name)

            for i, atoms in enumerate(tqdm(batch_structures)):
                config_group_name = f"config_{i}"
                config_group = batch_group.create_group(config_group_name)

                config_group.create_dataset('positions', data=atoms.positions)
                config_group.create_dataset('atomic_numbers', data=atoms.numbers)
                config_group.create_dataset('cell', data=atoms.cell.array)
                config_group.create_dataset('pbc', data=atoms.pbc)

                # add defaults for weights and types
                config_group.create_dataset('weight', data=np.array([1.0]))
                config_group.create_dataset('config_type', data='DFT'.encode('utf-8'))

                properties_group = config_group.create_group('properties')
                property_weights_group = config_group.create_group('property_weights')

                for key, value in atoms.arrays.items():
                    if key not in ['positions', 'numbers']:
                        properties_group.create_dataset(key, data=value)

                for key, value in atoms.info.items():
                    if isinstance(value, (int, float, np.number)):
                        properties_group.create_dataset(key, data=np.array([value]))
                    # try:
                    #     properties_group.attrs[key] = value
                    # except (TypeError, ValueError):
                    #     properties_group.attrs[key] = str(value)

    print(f"Successfully wrote {len(structures)} structures to HDF5 file")
else:
    print("No structures found! Check your file format.")

if len(structures) > 0:
    db = connect(str(file_name) + '.lmdb', type='aselmdb')
    for i, atoms in enumerate(tqdm(structures)):
        db.write(atoms)
        print(f"Successfully added structure_{i}")
else:
    print("No structures found! check your file format")

