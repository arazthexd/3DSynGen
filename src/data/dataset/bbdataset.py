import os, shutil, copy, time
from typing import List, Tuple, Dict, Optional, Any, Type
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

from mpire import WorkerPool

import numpy as np

from rdkit import Chem
from rdkit.Chem import (
    rdDistGeom, rdMolAlign, rdChemReactions, rdForceFieldHelpers, 
    rdFingerprintGenerator, Descriptors, Descriptors3D
)
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Pharm2D import Generate, DefaultSigFactory
from rdkit.Geometry import rdGeometry
from rdkit import rdBase
rdBase.DisableLog("rdApp.*")

from drfp import DrfpEncoder
from mordred import Calculator, Autocorrelation, Chi, EState

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors

import torch
from torch_geometric.data import Data, Dataset

from ...utils.dtypes import BuildingBlock as BB
from ...utils.helper import draw_cr, chi, cpsa, get_bond_edges, is_reactant

class AtomTypeFeaturizer:
    def __init__(self, 
                 types: List[str] = ["C", "N", "O", "X", "S", "P"],
                 defs: Dict[str, List[str]] = {"X": ["F", "Cl", "Br", "I"]}):
        self.types = types
        self.defs = defs

        self.symbol2type = {sym: typ 
                            for typ, syms in self.defs.items() for sym in syms}
        self.symbol2type.update({sym: sym for sym in self.types})

        self.onehot = OneHotEncoder(categories=[types], handle_unknown="ignore")
        
    def featurize_atoms(self, mol: Chem.Mol):
        atoms: List[Chem.Atom] = list(mol.GetAtoms())
        atom_types = [self.symbol2type.get(atom.GetSymbol(), "OTH") 
                      for atom in atoms]
        onehot_encoding = self.onehot.fit_transform(
            np.array(atom_types).reshape(-1, 1))
        return onehot_encoding.toarray()
    
class MoleculeRDKitFeaturizer:
    def __init__(self):
        self.include_func = lambda name: "Chi" in name

    def featurize_mol(self, mol: Chem.Mol):
        feats = []
        for name, func in Descriptors.descList:
            if self.include_func(name):
                feats.append(func(mol))
        return np.array(feats)

class BBFeaturizer:
    def __init__(self):
        self.atom_typer = AtomTypeFeaturizer()
        self.mol_rdfeaturizer = MoleculeRDKitFeaturizer()

        self.fscaler = StandardScaler()

    def featurize(self, mols: List[Chem.Mol], 
                  progbar: bool = True, 
                  transform_only: bool = False) -> torch.Tensor | List[Data]:

        data_list = []
        mol_feats = []
        for i, mol in enumerate(tqdm(mols, disable=not progbar)):
            atom_types = torch.tensor(self.atom_typer.featurize_atoms(mol))
            edge_index = torch.tensor(get_bond_edges(mol))
            data = Data(x=atom_types,
                        edge_index=edge_index)
            data_list.append(data)

            mol_feats.append(self.mol_rdfeaturizer.featurize_mol(mol))

        mol_feats = np.stack(mol_feats)
        if not transform_only:
            mol_feats = self.fscaler.fit_transform(mol_feats)
        else:
            mol_feats = self.fscaler.transform(mol_feats)
        mol_feats = torch.tensor(mol_feats)
        
        for i, data in enumerate(data_list):
            data.y = mol_feats[i].reshape(1, -1)
        
        return data_list
    
class BBConfFeaturizer:
    def __init__(self):
        pass

    def featurize(self, mols: List[Chem.Mol],
                  progbar: bool = True) -> torch.Tensor | List[Data]:
        
        def featurize_confs(mol: Chem.Mol):
            cmols = []
            for j in range(mol.GetNumConformers()):
                cmol = Chem.Mol(mol, confId=j)
                cmols.append(cmol)
            
            cmol_feats = torch.tensor([
                list(Descriptors3D.CalcMolDescriptors3D(mol).values())
                for mol in cmols
            ])
            
            cdata_list = []
            for i, cmol in enumerate(cmols):
                coords = cmol.GetConformer().GetPositions()
                cdata = Data(y=cmol_feats[i].reshape(1, -1),
                             pos=torch.tensor(coords))
                cdata_list.append(cdata)
            
            return cdata_list
        
        if len(mols) < 5:
            data_list = [featurize_confs(mol) for mol in mols]
        else:
            with WorkerPool(n_jobs=10) as pool:
                data_list = pool.map(featurize_confs, mols, 
                                        progress_bar=progbar)
        
        return data_list

class BBDataset(Dataset):
    def __init__(self, root: str = ".", 
                 bbs: List[BB] = list(), sdf_files: List[str] = list(),
                 featurizer: BBFeaturizer = BBFeaturizer(), 
                 feats: torch.Tensor | List[Data] = None,
                 cfeaturizer: BBConfFeaturizer = BBConfFeaturizer(),
                 cfeats: List[torch.Tensor | List[Data]] = None,
                 use_confs: bool = False,
                 force_reload: bool = False):
        
        if not os.path.exists(root):
            os.mkdir(root)
        
        self.bbs = bbs
        self._input_sdfs = sdf_files
        self.featurizer = featurizer
        self.feats = feats
        self.cfeaturizer = cfeaturizer
        self.cfeats = cfeats
        self.use_confs = use_confs

        super().__init__(root, force_reload=force_reload)

        self.bbs = torch.load(self.processed_paths[0])
        self.feats = torch.load(self.processed_paths[1])
        if use_confs:
            self.cfeats = torch.load(self.processed_paths[2])

    @property 
    def raw_file_names(self): 
        return self._input_sdfs
    
    @property 
    def processed_file_names(self):
        if not self.use_confs:
            return ["bbs.pt", "bbfeats.pt"]
        return ["bbs.pt", "bbfeats.pt", "bbconffeats.pt"] 
    
    def download(self): 
        sdf_names = []
        for sdf in self._input_sdfs:
            sdf_name = os.path.basename(sdf)
            if not os.path.exists(os.path.join(self.root, sdf_name)):
                shutil.copy(sdf, self.root)
            sdf_names.append(sdf_name) 
    
    def process(self):
        time.sleep(0.5)

        print("Loading building blocks...")
        bbs = self.bbs.copy()
        for sdf in self._input_sdfs:
            bbs.extend(BB.read_from_sdf(sdf))

        print("Creating conformers for building blocks...")
        bbs = self.create_conformers(bbs, n=2)
        torch.save(bbs, open(self.processed_paths[0], 'wb'))
        
        if self.feats is None:
            if self.featurizer is None:
                raise ValueError("Either featurizer or feats must be provided")
            
            print("Featurizing building blocks...")
            feats = self.featurizer.featurize(bbs)
            torch.save(feats, open(self.processed_paths[1], 'wb'))

            if self.use_confs:
                print("Featurizing building block conformers...")
                cfeats = self.cfeaturizer.featurize(bbs)
                torch.save(cfeats, open(self.processed_paths[2], 'wb'))

    def len(self):
        return len(self.bbs)

    def get(self, idx):
        return self.feats[idx]
    
    def get_bb(self, idx) -> BB:
        return self.bbs[idx]
    
    def cget(self, idx, cidx):
        return self.cfeats[idx][cidx]
    
    @staticmethod
    def create_conformers(bbs: List[BB], n: int = 1) -> List[BB]:
        def create_conformer(bb):
            bb = copy.copy(bb)
            rdDistGeom.EmbedMultipleConfs(bb, n)
            return bb
        
        with WorkerPool(n_jobs=10) as pool:
            newbbs = pool.map(create_conformer, bbs, progress_bar=True)

        return newbbs
    
    @staticmethod
    def merge_feats(data1: Data, data2: Data) -> Data:

        def merge_values(key, v1: torch.Tensor, v2: torch.Tensor):
            # key taken for potential exceptions
            return torch.cat((v1, v2), dim=1)
            
        dict1 = data1.to_dict()
        dict2 = data2.to_dict()

        merged_dict = {}
        for key in dict1: 
            if key in dict2: 
                merged_dict[key] = merge_values(key, dict1[key], dict2[key]) 
            else: 
                merged_dict[key] = dict1[key]

        for key in dict2: 
            if key not in dict1: 
                merged_dict[key] = dict2[key]

        return Data.from_dict(merged_dict)