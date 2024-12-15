import os
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

sys.path.append('3D-MIL-QSSR/miqssr')
from conformer_generation import gen_conformers
from conformer_generation.psearch_master import (gen_conf_rdkit,
                                                 gen_stereo_rdkit)
from descriptor_calculation.pmapper_3d import convert_pkl_to_sdf


def gen_confs(fname, nconfs_list, energy=50, path=None, ncpu=4, verbose=True):
    '''
    :param fname: smi file. Mol_name, smiles, act
    :param nconfs_list: list[int]
    :param path: out path. If None uses dirname of fname
    :param ncpu: int
    :return:
    '''

    if path is None:
        path = os.path.dirname(fname)
    if not os.path.exists(path):
        os.makedirs(path)

    in_fname = fname

    #print('Conformers generation')

    max_conf = max(nconfs_list)

    conf_log = os.path.join(path, 'conformers_log.pkl')
    # conf_tupl[-1]=energy
    gen_conf_rdkit.main_params(in_fname=in_fname,
                               out_fname=conf_log,
                               id_field_name=None,
                               nconf=max_conf,
                               energy=energy,
                               rms=.5,
                               ncpu=ncpu,
                               seed=42,
                               verbose=verbose,
                               log=True)

    out_partfname = os.path.join(path, 'conformers.pkl')
    # take n-confromer from conformer-log file
    out_fnames = gen_conformers.get_n_confs(conf_log=conf_log,
                                            nconf_list=nconfs_list,
                                            out_partfname=out_partfname)

    convert_pkl_to_sdf(out_partfname, os.path.join(path, 'conformers.sdf'))

    return os.path.join(path, 'conformers.sdf')


def make_csv_rotatable_bond_thresholding(data_path, save_dir):
    '''filter by rotatable_bonds >= 6
    '''
    df = pd.read_csv(data_path)
    df_filtered = df[df['rotatable_bonds'] >= 6]
    save_path = os.path.join(save_dir, 'smiles_rotatable_bonds_6.csv')
    df_filtered.to_csv(save_path, index=False)


def gen_confs_threshold6(data_path,
                         num_confs=[50],
                         energy=50,
                         num_cpu=1,
                         path='.'):
    '''
    generate conformer that compound have rotatable_bonds >=6
    '''
    df = pd.read_csv(data_path, usecols=['smiles', 'csid'])
    #
    res = []
    for _, row in df.iterrows():
        res.append({
            'SMILES': row['smiles'],
            'MOL_ID': row['csid'],
            'ACT': None
        })

    #
    cat_file = os.path.join(path, 'smiles.smi')
    res = pd.DataFrame(res).drop_duplicates()
    res.to_csv(cat_file, index=False, header=False)
    #
    sdf_path = gen_confs(cat_file,
                         nconfs_list=[num_confs],
                         energy=energy,
                         ncpu=num_cpu,
                         path=path,
                         verbose=False)

    return sdf_path


if __name__ == "__main__":
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)
    # make csv rotatable bond => 6
    DATA_PATH = 'xxx'
    make_csv_rotatable_bond_thresholding(DATA_PATH, SAVE_DIR)
    # generate conformers
    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)
    gen_confs_threshold6(DATA_PATH,
                         num_confs=50,
                         energy=50,
                         num_cpu=10,
                         path=SAVE_DIR)
