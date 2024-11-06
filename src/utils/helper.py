import os
from typing import List, Tuple, Dict, Optional, Any, Iterable
from tqdm import tqdm
import dill
import random
from functools import partial
import warnings
warnings.simplefilter("ignore")

from IPython.display import SVG

import numpy as np
import pandas as pd
import seaborn as sns

from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage, rdMolDraw2D
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Pharm2D import Generate, DefaultSigFactory
from rdkit.Chem import rdChemReactions
from rdkit import rdBase
rdBase.DisableLog("rdApp.*")

from mordred import Calculator, Autocorrelation, Chi, EState, CPSA

from sklearn.decomposition import PCA

from .dtypes import BuildingBlock as BB
from .dtypes import CustomReaction as CR

def show_bbs(bbs: List[BB], n=10, npr=5, rand=True):
    if rand:
        idx = random.sample(list(range(len(bbs))), n)
    else:
        idx = list(range(n))
    return MolsToGridImage([bbs[i].rdmol for i in idx], 
                           molsPerRow=npr, subImgSize=(250, 250),
                           legends=[str(i) for i in idx])

def draw_cr(reaction: CR):
    drawer = rdMolDraw2D.MolDraw2DSVG(900, 300)
    dopts = drawer.drawOptions()
    rdChemReactions.Compute2DCoordsForReaction(reaction)
    dopts.bondLineWidth = 1.5 # default is 2.0
    drawer.DrawReaction(reaction)
    drawer.FinishDrawing()
    return SVG(data=drawer.GetDrawingText().replace('svg:',''))

def visualize_featurized(ax, bbs, featurize_func, hue_func=None, n=None):
    if n is None:
        n = len(bbs)
    mols = [bb.rdmol for bb in bbs[:n]]
    feats = featurize_func(mols) # numpy array
    if hue_func is None:
        hues = None
    else:
        hues = hue_func(mols)
    pca = PCA(2)
    feats = pca.fit_transform(feats)
    sns.scatterplot(x=feats[:, 0], y=feats[:, 1], ax=ax, hue=hues)

def autocorrelation(mols):
    calc = Calculator(Autocorrelation)
    df = calc.pandas(mols)
    df = df.apply(partial(pd.to_numeric, errors="coerce")).dropna(axis=1)
    return df.values

def chi(mols, progbar = True):
    calc = Calculator(Chi)
    df = calc.pandas(mols, quiet=not progbar)
    df = df.apply(partial(pd.to_numeric, errors="coerce")).dropna(axis=1)
    return df.values

def estate(mols):
    calc = Calculator(EState)
    df = calc.pandas(mols)
    df = df.apply(partial(pd.to_numeric, errors="coerce")).dropna(axis=1)
    return df.values

def cpsa(mols):
    calc = Calculator(CPSA)
    df = calc.pandas(mols, quiet=True, nproc=1)
    df = df.apply(partial(pd.to_numeric, errors="coerce")).dropna(axis=1)
    return df.values

def morganfp(mols, radius=3):
    fpgen = GetMorganGenerator(radius)
    return np.array(fpgen.GetFingerprints(mols))

def rdpharm2d(mols):
    sigfactory = DefaultSigFactory(bins=[(0,2),(2,5),(5,8)])
    return np.array([Generate.Gen2DFingerprint(m, sigfactory) for m in mols])

def is_reactant(mols, rxn: CR, ridx=None):
    rs = [rxn.is_reactant(mol) for mol in mols]
    if ridx is None:
        return [len(r)>0 for r in rs]
    else:
        return [ridx in r for r in rs]
    
def get_bond_edges(mol: Chem.Mol) -> np.ndarray:

    bonds: Iterable[Chem.Bond] = mol.GetBonds()
    bond_edges_1 = [bond.GetBeginAtomIdx() for bond in bonds] + \
        [bond.GetEndAtomIdx() for bond in bonds]
    bond_edges_2 = [bond.GetEndAtomIdx() for bond in bonds] + \
        [bond.GetBeginAtomIdx() for bond in bonds]
    bond_edges = np.array([bond_edges_1, bond_edges_2])
    return bond_edges