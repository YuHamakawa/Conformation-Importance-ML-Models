"""
This script generates Mol objects from atomic numbers and 3D coordinates,
processes the molecules, and outputs them to an SDF file.
# cite "https://zenn.dev/hodakam/articles/ad5fce800a6cd2"
"""

import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_mol_object(row: pd.Series) -> Chem.Mol:
    """
    Generate a Mol object from atomic numbers and 3D coordinates.

    Args:
        row (pd.Series): A row from the input dataframe.

    Returns:
        Chem.Mol: The generated Mol object.
    """
    # Extract atomic numbers and coordinates from the row
    atomic_numbers = list(
        map(int,
            row['pubchem.PM6.atoms.elements.number'].strip('[]').split(', ')))
    coordinates = list(
        map(float, row['pubchem.PM6.atoms.coords.3d'].strip('[]').split(', ')))
    assert len(atomic_numbers) * 3 == len(coordinates)

    # This doesn't work well...
    #smiles = row['pubchem.PM6.openbabel.Canonical SMILES']
    # Generate Mol object from SMILES
    smiles = row['pubchem.Isomeric SMILES']
    mol = Chem.MolFromSmiles(smiles)

    # Remove molecules with structural issues
    if mol is None:
        return None

    # Add explicit hydrogens (e.g., deuterium)
    # mol = Chem.AddHs(mol, explicitOnly=True, addCoords=True)

    # Generate conformer: Embed the molecule in 3D space.
    if AllChem.EmbedMolecule(mol, randomSeed=42):
        print(f"Failed to generate Conformer for CID: {row['pubchem.cid']}")
        return None

    # Get the conformer of the molecule
    conf = mol.GetConformer(0)

    # Remove all hydrogen atoms from the atomic numbers and coordinates
    atomic_numbers_without_hydrogen = [
        element for element in atomic_numbers if element != 1
    ]
    coordinates_without_hydrogen = [
        coordinates[i:i + 3] for i in range(0, len(coordinates), 3)
        if atomic_numbers[i // 3] != 1
    ]

    # Check if there are hydrogen atoms to be deleted
    delete_frag = False
    # Delete molecules containing hydrogen in the SDF file (e.g., deuterium),
    # since the correspondence with coordinates is unknown.
    for _, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            delete_frag = True

    if delete_frag:
        return None

    # Set the atom positions in the conformer
    for i, _ in enumerate(atomic_numbers_without_hydrogen):
        x, y, z = coordinates_without_hydrogen[i]
        conf.SetAtomPosition(i, (x, y, z))

    # Set the CID as a property of the molecule
    mol.SetProp('_Name', str(row['pubchem.cid']))

    # Add the 'RotatableBonds' column as a property of the Mol object
    # mol.SetIntProp('RotatableBonds', row['RotatableBonds'])
    # print(Chem.MolToMolBlock(mol))

    return mol


def process_molecules(data_path: str, save_path: str) -> None:
    """
    Read a CSV file, process the molecules, and output them to an SDF file.

    Args:
        data_path (str): The path to the CSV file.
        save_path (str): The path to save the SDF file.
    """

    df = pd.read_csv(data_path)

    #df = df.head(10) # for small test

    mol_list = [generate_mol_object(row) for _, row in df.iterrows()]
    # remove invalit molecute (i,e, None object)
    mol_list = [mol for mol in mol_list if mol is not None]

    print(f'read molecule: {df.shape[0]}\nscan molecule: {len(mol_list)}')

    # output to SDF file
    writer = Chem.SDWriter(save_path)

    for mol in mol_list:
        writer.write(mol)

    writer.close()


if __name__ == '__main__':

    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    SAVE_PATH = os.path.join(SAVE_DIR, 'CHON500noSalt_rotatable_top100k.sdf')

    process_molecules(DATA_PATH, SAVE_PATH)
