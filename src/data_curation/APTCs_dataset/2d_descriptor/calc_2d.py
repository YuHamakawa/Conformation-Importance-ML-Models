import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate, SigFactory


def _smiles_to_ecfp(smiles, which_fp, radius=2, nBits=1024):
    '''
    Convert SMILES to ECFP or FCFP
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        if which_fp == 'fcfp_count':
            ecfp = AllChem.GetHashedMorganFingerprint(
                mol,
                radius=radius,
                nBits=nBits,
                invariants=[1] * mol.GetNumAtoms(),  # fcfp
            )
            # NumPy配列に変換
            ecfp_arr = np.zeros((nBits, ), dtype=int)
            for bit_id, count in ecfp.GetNonzeroElements().items():
                if bit_id < nBits:  # ビットIDが範囲内にあることを確認
                    ecfp_arr[bit_id] = count
                else:
                    print(f'bit_id {bit_id} is out of range')
        elif which_fp == 'fcfp_bit':
            ecfp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=radius,
                nBits=nBits,
                invariants=[1] * mol.GetNumAtoms(),  # fcfp
            )
            # Convert the ECFP to a numpy array
            ecfp_arr = np.array(ecfp)

        elif which_fp == 'ecfp_count':
            ecfp = AllChem.GetHashedMorganFingerprint(
                mol,
                radius=radius,
                nBits=nBits,
            )
            # NumPy配列に変換
            ecfp_arr = np.zeros((nBits, ), dtype=int)
            for bit_id, count in ecfp.GetNonzeroElements().items():
                if bit_id < nBits:  # ビットIDが範囲内にあることを確認
                    ecfp_arr[bit_id] = count
                else:
                    print(f'bit_id {bit_id} is out of range')
        elif which_fp == 'ecfp_bit':
            ecfp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=radius,
                nBits=nBits,
            )
            # Convert the ECFP to a numpy array
            ecfp_arr = np.array(ecfp)
        else:
            raise ValueError('which_fp is invalid')

        return ecfp_arr


def calc_ecfp(save_dir, which_fp):
    '''
    calc ecfp from SMILES
    '''
    training_dataset_path_1 = '3D-MIL-QSSR/datasets/aptc-1/aptc_1_training_datatest.csv'
    test_dataset_path_1 = '3D-MIL-QSSR/datasets/aptc-1/aptc_1_test_datatest.csv'
    training_dataset_path_2 = '3D-MIL-QSSR/datasets/aptc-2/aptc_2_training_datatest.csv'
    test_dataset_path_2 = '3D-MIL-QSSR/datasets/aptc-2/aptc_2_test_datatest.csv'

    train_data_1 = pd.read_csv(training_dataset_path_1,
                               usecols=['CATALYST', 'ACTIVITY'])
    test_data_1 = pd.read_csv(test_dataset_path_1,
                              usecols=['CATALYST', 'ACTIVITY'])

    train_data_2 = pd.read_csv(training_dataset_path_2,
                               usecols=['CATALYST', 'ACTIVITY'])
    test_data_2 = pd.read_csv(test_dataset_path_2,
                              usecols=['CATALYST', 'ACTIVITY'])

    train_data_1['ECFP'] = train_data_1['CATALYST'].apply(_smiles_to_ecfp,
                                                          which_fp=which_fp)
    test_data_1['ECFP'] = test_data_1['CATALYST'].apply(_smiles_to_ecfp,
                                                        which_fp=which_fp)
    train_data_2['ECFP'] = train_data_2['CATALYST'].apply(_smiles_to_ecfp,
                                                          which_fp=which_fp)
    test_data_2['ECFP'] = test_data_2['CATALYST'].apply(_smiles_to_ecfp,
                                                        which_fp=which_fp)

    # save data
    save_dir = os.path.join(save_dir, which_fp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_data_1.to_pickle(
        os.path.join(save_dir, f'aptc1_{which_fp}_train.pkl'))
    test_data_1.to_pickle(os.path.join(save_dir, f'aptc1_{which_fp}_test.pkl'))
    train_data_2.to_pickle(
        os.path.join(save_dir, f'aptc2_{which_fp}_train.pkl'))
    test_data_2.to_pickle(os.path.join(save_dir, f'aptc2_{which_fp}_test.pkl'))


def _smiles_to_pharm2d(smiles):
    '''
    Calculate pharmacophore fingerprint
    '''
    # custom feature definion based on Pmapper definition
    custom_features = """
    DefineFeature AliphaAtom [C,N,O,S,P,F,Cl,Br,I]
    Family AliphaAtom
    Weights 1.0
    EndFeature

    AtomType AromR5 [a;r5,!R1&r4,!R1&r3]
    DefineFeature Arom5 [{AromR5}]1:[{AromR5}]:[{AromR5}]:[{AromR5}]:[{AromR5}]:1
    Family Aromatic
    Weights 1.0,1.0,1.0,1.0,1.0
    EndFeature

    AtomType AromR6 [a;r6,!R1&r5,!R1&r4,!R1&r3]
    DefineFeature Arom6 [{AromR6}]1:[{AromR6}]:[{AromR6}]:[{AromR6}]:[{AromR6}]:[{AromR6}]:1
    Family Aromatic
    Weights 1.0,1.0,1.0,1.0,1.0,1.0
    EndFeature
    """

    #fdef_base = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    feats = ChemicalFeatures.BuildFeatureFactoryFromString(custom_features)

    # ATPC1 catalysts example
    # test_smiles = 'CCC1C[N+]2(CCC=C)CCC1CC2C(OCCC=C)C1=C2C=CC=CC2=NC=C1'
    mol = Chem.MolFromSmiles(smiles)

    # pharmacophore fingerprint generation
    fp_factory = SigFactory.SigFactory(
        featFactory=feats,
        useCounts=True,
        minPointCount=3,  # only 3 points pharmacophores
        maxPointCount=3,
        shortestPathsOnly=True,
        includeBondOrder=False,
        trianglePruneBins=False)

    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    fp_factory.SetBins(bins)
    fp_factory.Init()  # just in case

    fp = Generate.Gen2DFingerprint(mol, fp_factory)

    fc_array = np.array(fp.ToList())

    # for interpretation, see
    # https://www.rdkit.org/docs/RDKit_Book.html#representation-of-pharmacophore-fingerprints

    # bitdict = fp.GetNonzeroElements()
    # for bit, count in bitdict.items():
    #     desc = fp_factory.GetBitDescription(bit)
    #     binfo = fp_factory.GetBitInfo(bit)
    #     print(
    #         f'Bit: {bit} '
    #         f'Count: {count} '
    #         f'Description: {desc} '
    #         f'Bitinfo: {binfo} ',
    #         sep='\t')

    # GetBitInfo format
    # 1 the number of points in the pharmacophore
    # 2 the proto-pharmacophore (tuple of pattern indices)
    # 3 the scaffold (tuple of distance indices)
    # Bit description format
    # pharmacophore, distance matrix

    return fc_array


def calc_pharm2d(save_dir):
    training_dataset_path_1 = '3D-MIL-QSSR/datasets/aptc-1/aptc_1_training_datatest.csv'
    test_dataset_path_1 = '3D-MIL-QSSR/datasets/aptc-1/aptc_1_test_datatest.csv'
    training_dataset_path_2 = '3D-MIL-QSSR/datasets/aptc-2/aptc_2_training_datatest.csv'
    test_dataset_path_2 = '3D-MIL-QSSR/datasets/aptc-2/aptc_2_test_datatest.csv'

    train_data_1 = pd.read_csv(training_dataset_path_1,
                               usecols=['CATALYST', 'ACTIVITY'])
    test_data_1 = pd.read_csv(test_dataset_path_1,
                              usecols=['CATALYST', 'ACTIVITY'])

    train_data_2 = pd.read_csv(training_dataset_path_2,
                               usecols=['CATALYST', 'ACTIVITY'])
    test_data_2 = pd.read_csv(test_dataset_path_2,
                              usecols=['CATALYST', 'ACTIVITY'])

    train_data_1['ECFP'] = train_data_1['CATALYST'].apply(_smiles_to_pharm2d)
    test_data_1['ECFP'] = test_data_1['CATALYST'].apply(_smiles_to_pharm2d)
    train_data_2['ECFP'] = train_data_2['CATALYST'].apply(_smiles_to_pharm2d)
    test_data_2['ECFP'] = test_data_2['CATALYST'].apply(_smiles_to_pharm2d)

    # save data
    save_dir = os.path.join(save_dir, 'pharm_2d')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_data_1.to_pickle(os.path.join(save_dir, f'aptc1_pharm_2d_train.pkl'))
    test_data_1.to_pickle(os.path.join(save_dir, f'aptc1_pharm_2d_test.pkl'))
    train_data_2.to_pickle(os.path.join(save_dir, f'aptc2_pharm_2d_train.pkl'))
    test_data_2.to_pickle(os.path.join(save_dir, f'aptc2_pharm_2d_test.pkl'))


def test_smiles_to_pharm2d():
    '''
    Calculate pharmacophore fingerprint
    confirm the bit infomation
    '''
    # custom feature definion based on Pmapper definition
    custom_features = """
    DefineFeature AliphaAtom [C,N,O,S,P,F,Cl,Br,I]
    Family AliphaAtom
    Weights 1.0
    EndFeature

    AtomType AromR5 [a;r5,!R1&r4,!R1&r3]
    DefineFeature Arom5 [{AromR5}]1:[{AromR5}]:[{AromR5}]:[{AromR5}]:[{AromR5}]:1
    Family Aromatic
    Weights 1.0,1.0,1.0,1.0,1.0
    EndFeature

    AtomType AromR6 [a;r6,!R1&r5,!R1&r4,!R1&r3]
    DefineFeature Arom6 [{AromR6}]1:[{AromR6}]:[{AromR6}]:[{AromR6}]:[{AromR6}]:[{AromR6}]:1
    Family Aromatic
    Weights 1.0,1.0,1.0,1.0,1.0,1.0
    EndFeature
    """

    #fdef_base = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    feats = ChemicalFeatures.BuildFeatureFactoryFromString(custom_features)
    test_smiles = 'CCC1C[N+]2(CCC=C)CCC1CC2C(OCCC=C)C1=C2C=CC=CC2=NC=C1'
    # ATPC1 catalysts example
    mol = Chem.MolFromSmiles(test_smiles)

    # pharmacophore fingerprint generation
    fp_factory = SigFactory.SigFactory(
        featFactory=feats,
        useCounts=True,
        minPointCount=3,  # only 3 points pharmacophores
        maxPointCount=3,
        shortestPathsOnly=True,
        includeBondOrder=False,
        trianglePruneBins=False)

    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    fp_factory.SetBins(bins)
    fp_factory.Init()  # just in case

    fp = Generate.Gen2DFingerprint(mol, fp_factory)

    fc_array = np.array(fp.ToList())

    # for interpretation, see
    # https://www.rdkit.org/docs/RDKit_Book.html#representation-of-pharmacophore-fingerprints

    bitdict = fp.GetNonzeroElements()
    for bit, count in bitdict.items():
        desc = fp_factory.GetBitDescription(bit)
        binfo = fp_factory.GetBitInfo(bit)
        print(
            f'Bit: {bit} '
            f'Count: {count} '
            f'Description: {desc} '
            f'Bitinfo: {binfo} ',
            sep='\t')

    # GetBitInfo format
    # 1 the number of points in the pharmacophore
    # 2 the proto-pharmacophore (tuple of pattern indices)
    # 3 the scaffold (tuple of distance indices)
    # Bit description format
    # pharmacophore, distance matrix

    return fc_array


if __name__ == '__main__':
    SAVE_DIR = 'xxx'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    calc_pharm2d(SAVE_DIR)
    test_smiles_to_pharm2d()

    fp_list = ['ecfp_bit', 'fcfp_bit', 'ecfp_count', 'fcfp_count']
    for which_fp in fp_list:
        calc_ecfp(SAVE_DIR, which_fp)
