"""
241113 added, Yu Hamakawa.
Processing of MP dataset.
"""

import os
from os.path import exists, join

import numpy as np
import pandas as pd
from pahelix.datasets.inmemory_dataset import InMemoryDataset
from rdkit import Chem


def get_default_mp_task_names():
    """Get that default freesolv task names and return measured expt"""
    return ['mpC']


def load_mp_dataset(data_path, task_names=None, test_mode=False):
    """
    tbd
    """
    if task_names is None:
        task_names = get_default_mp_task_names()

    supplier = Chem.SDMolSupplier(data_path, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]

    data_list = []
    for mol in mols:
        labels = []
        for task in task_names:
            label = float(mol.GetProp(task))
            labels.append(label)

        data = {
            'mol': mol,
            'label': np.array(labels),
        }
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset


def get_mp_stat(data_path, task_names, test_mode=False):
    """Return mean and std of labels"""

    supplier = Chem.SDMolSupplier(data_path, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]

    mean_list = []
    std_list = []
    for task in task_names:
        labels = []
        for mol in mols:
            label = float(mol.GetProp(task))
            labels.append(label)

        mean_list.append(np.mean(labels))
        std_list.append(np.std(labels))

    return {
        'mean': np.array(mean_list),
        'std': np.array(std_list),
        'N': len(mols),
    }
