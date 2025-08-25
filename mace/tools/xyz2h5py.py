from ase.db import connect
from ase.io.extxyz import read_extxyz
import h5py
 
with open('force_trajectory.xyz', 'r') as f:  
    structures = list(read_extxyz(f))
 
with h5py.File('pt4sn.h5', 'w') as z:
    for i, atoms in enumerate(structures):
        group_name = f"structure_{i}"
        structure_group = z.create_group(group_name)
 
        structure_group.create_dataset('positions', data =atoms.positions)
        structure_group.create_dataset('numbers', data=atoms.numbers)
        structure_group.create_dataset('cell', data=atoms.cell.array)
        structure_group.create_dataset('pbc', data=atoms.pbc)
 
        for key, value in atoms.arrays.items():
            if key not in ['positions', 'numbers']:
                structure_group.create_dataset(key, data=value)
 
        for key, value in atoms.info.items():
            structure_group.attrs[key] = value