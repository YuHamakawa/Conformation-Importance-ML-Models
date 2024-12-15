import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import CalcNumAtomStereoCenters


def detect_invalid_smiles(df):
    invalid_smiles = []
    for _, row in df.iterrows():
        smiles = row['smiles']
        csid = row['csid']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append((csid, smiles))
    return invalid_smiles


def check_undefined_stereo(mol):
    """
    Function to determine if a molecule has undefined chirality or cis-trans isomerism information.
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol): RDKit Mol object
    
    Returns:
    bool: True if the molecule has undefined chirality centers or undefined cis-trans isomerism information
    """
    if mol is None:
        return False

    # Get chiral centers (including undefined)
    stereo_centers = Chem.FindMolChiralCenters(mol,
                                               includeUnassigned=True,
                                               useLegacyImplementation=False)
    undefined_chirality = any(chirality == '?'
                              for _, chirality in stereo_centers)

    # Check for cis-trans isomerism
    undefined_cis_trans = False
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo == Chem.rdchem.BondStereo.STEREONONE:
                # Stereochemistry of the double bond is undefined
                # Additionally, check the substituents around the double bond to determine if E/Z is possible
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()

                # Count the number of different groups attached to each carbon
                begin_neighbors = [
                    nbr.GetIdx() for nbr in begin_atom.GetNeighbors()
                    if nbr.GetIdx() != end_atom.GetIdx()
                ]
                end_neighbors = [
                    nbr.GetIdx() for nbr in end_atom.GetNeighbors()
                    if nbr.GetIdx() != begin_atom.GetIdx()
                ]

                if len(begin_neighbors) > 1 and len(end_neighbors) > 1:
                    undefined_cis_trans = True
                    break

    return undefined_chirality or undefined_cis_trans


def extract_mols_with_undefined_stereo(df):
    """
    Function to extract molecules with undefined chirality or cis-trans isomerism information from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing RDKit Mol objects in the 'mol' column
    
    Returns:
    pd.DataFrame: New DataFrame containing molecules with undefined chirality or cis-trans isomerism information
    """
    undefined_stereo_rows = df[df['mol'].apply(
        check_undefined_stereo)].reset_index(drop=True)
    return undefined_stereo_rows


def has_chiral_centers(mol):
    """
    Function to determine if a molecule has chiral centers.
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol): RDKit Mol object
    
    Returns:
    bool: True if the molecule has chiral centers
    """
    stereo_centers = Chem.FindMolChiralCenters(mol,
                                               includeUnassigned=True,
                                               useLegacyImplementation=False)
    return len(stereo_centers) > 0


def has_cis_trans_isomers(mol):
    """
    Function to determine if a molecule has double bonds that could exhibit cis-trans isomerism.
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol): RDKit Mol object
    
    Returns:
    bool: True if the molecule has double bonds that could exhibit cis-trans isomerism
    """
    for bond in mol.GetBonds():
        if bond.GetBondType(
        ) == Chem.rdchem.BondType.DOUBLE and not bond.GetIsAromatic():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            # Count the number of different groups attached to each carbon (e.g., ethyl groups)
            begin_neighbors = [
                nbr for nbr in begin_atom.GetNeighbors()
                if nbr.GetIdx() != end_atom.GetIdx()
            ]
            end_neighbors = [
                nbr for nbr in end_atom.GetNeighbors()
                if nbr.GetIdx() != begin_atom.GetIdx()
            ]

            if len(begin_neighbors) > 1 and len(end_neighbors) > 1:
                return True
    return False


def has_possible_stereo(mol):
    """
    Function to determine if a molecule has chiral centers or double bonds that could exhibit cis-trans isomerism.
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol): RDKit Mol object
    
    Returns:
    bool: True if the molecule has chiral centers or double bonds that could exhibit cis-trans isomerism
    """
    if mol is None:
        return False
    return has_chiral_centers(mol) or has_cis_trans_isomers(mol)


def extract_mols_with_possible_stereo(df):
    """
    Function to extract molecules with chiral centers or double bonds that could exhibit cis-trans isomerism from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing RDKit Mol objects in the 'mol' column
    
    Returns:
    pd.DataFrame: New DataFrame containing molecules with chiral centers or double bonds that could exhibit cis-trans isomerism
    """
    possible_stereo_rows = df[df['mol'].apply(
        has_possible_stereo)].reset_index(drop=True)
    return possible_stereo_rows


def step1_valid_smiles(data_path, save_dir):
    df = pd.read_csv(data_path)
    # Check for duplicate data.
    check_columns = ['key', 'name', 'smiles', 'mpC', 'csid']
    for col in check_columns:
        duplicates = df[df[col].duplicated(
            keep=False)]  # keep=False to get all duplicate rows
        if not duplicates.empty:
            print(f"Duplicate data in column '{col}':")
            print(duplicates[[
                col
            ]])  # Display duplicate data (only the relevant column)
        else:
            print(f"No duplicates in column '{col}'.")

    # Check for duplicates in the 'name' column
    print('Number of names:', len(df['name']))
    print('Number of unique names:', len(df['name'].unique()))
    # Display data with duplicate names
    tmp = df[df.duplicated(subset='name', keep=False)].sort_values('name')
    print(tmp)

    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df['rotatable_bonds'] = df['mol'].apply(
        lambda mol: Lipinski.NumRotatableBonds(mol) if mol else None)
    df['heavy_atoms'] = df['mol'].apply(lambda mol: mol.GetNumHeavyAtoms()
                                        if mol else None)

    # Check for missing values
    print(df.isnull().sum())

    # Detect invalid SMILES
    # invalid_smiles = detect_invalid_smiles(df)
    # undefined_stereo_df = extract_mols_with_undefined_stereo(df)
    possible_stereo_df = extract_mols_with_possible_stereo(df)

    df = df[~df['csid'].isin(possible_stereo_df['csid'])].reset_index(
        drop=True)

    # delete nan
    df = df.dropna(subset=['mol']).reset_index(drop=True)
    # delete Mol
    df.drop(columns=['mol'], inplace=True)

    df.to_csv(f'{save_dir}/valid_smiles.csv', index=False)


def plot_dist():
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)
    sns.histplot(df['rotatable_bonds'])
    plt.title('Distribution of Rotatable Bonds')
    plt.xlabel('Number of Rotatable Bonds')
    plt.ylabel('Frequency')
    plt.savefig(f'{SAVE_DIR}/rotatable_bonds.png')
    plt.close()

    sns.histplot(df['heavy_atoms'])
    plt.title('Distribution of Heavy Atoms')
    plt.xlabel('Number of Heavy Atoms')
    plt.ylabel('Frequency')
    plt.savefig(f'{SAVE_DIR}/heavy_atoms.png')
    plt.close()

    # Display statistics and distribution of mpC
    print(df['mpC'].describe())
    # sns.histplot(df['mpC'], bins=20, kde=True)
    # plt.show()

    # Display statistics and distribution of count
    print(df['count'].describe())
    # sns.histplot(df['count'], bins=20, kde=True)
    # plt.show()

    # Display statistics and distribution of range
    print(df['range'].describe())

    # sns.histplot(df['range'], bins=20, kde=True)
    # plt.show()


if __name__ == "__main__":
    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)

    step1_valid_smiles(DATA_PATH, SAVE_DIR)
