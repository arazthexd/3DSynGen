import os, shutil, copy, time
from typing import List, Tuple, Dict, Optional, Any, Type
from tqdm import tqdm
import dill
import random
from functools import partial
import warnings
warnings.simplefilter("ignore")

from mpire import WorkerPool

from rdkit import Chem
from rdkit.Chem import (
    rdDistGeom, rdMolAlign, rdChemReactions, rdForceFieldHelpers, 
    rdFingerprintGenerator, Descriptors
)
from rdkit import rdBase
rdBase.DisableLog("rdApp.*")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors

import torch
from torch_geometric.data import Data, Dataset

from ...utils.dtypes import CustomReaction as CR

class CRFeaturizer:
    def __init__(self):
        self.fp_params = rdChemReactions.ReactionFingerprintParams()
        self.fp_params.fpSize = 1024

    def featurize(self, 
                  rxns: List[rdChemReactions.ChemicalReaction]) \
                    -> torch.Tensor | List[Data]:
        time.sleep(0.1)

        feats = []
        for i, rxn in enumerate(tqdm(rxns)):
            rxn_feats = rdChemReactions.CreateDifferenceFingerprintForReaction(
                rxn, self.fp_params).ToList()
            feats.append(rxn_feats)
        
        return torch.tensor(feats)

class CRDataset(Dataset):
    def __init__(self, root: str = ".", 
                 rxns: List[CR] = list(), rxn_files: List[str] = list(),
                 featurizer: CRFeaturizer = CRFeaturizer(), 
                 feats: torch.Tensor | List[Data] = None,
                 force_reload: bool = False):
        
        if not os.path.exists(root):
            os.mkdir(root)
        
        self.rxns = rxns
        self._input_rxns = rxn_files
        self.featurizer = featurizer
        self.feats = feats

        super().__init__(root, force_reload=force_reload)

        self.rxns = torch.load(self.processed_paths[0])
        self.feats = torch.load(self.processed_paths[1])

    @property 
    def raw_file_names(self): 
        return self._input_rxns
    
    @property 
    def processed_file_names(self): 
        return ["rxns.pt", "rxnfeats.pt"] 
    
    def download(self): 
        rxn_names = []
        for rxn_file in self._input_rxns:
            rxn_name = os.path.basename(rxn_file)
            if not os.path.exists(os.path.join(self.root, rxn_name)):
                shutil.copy(rxn_file, self.root)
            rxn_names.append(rxn_name) # ?
    
    def process(self):
        time.sleep(1)

        print("Loading reactions...")
        rxns = self.rxns.copy()
        for rxn_file in self._input_rxns:
            rxns.extend(CR.parse_txt(rxn_file))
        torch.save(rxns, open(self.processed_paths[0], 'wb'))
        
        if self.feats is None:
            if self.featurizer is None:
                raise ValueError("Either featurizer or feats must be provided")
            
            print("Featurizing reactions...")
            feats = self.featurizer.featurize(rxns)
            torch.save(feats, open(self.processed_paths[1], 'wb'))

    def len(self):
        return len(self.rxns)

    def get(self, idx):
        return self.feats[idx]
    
    def get_rxn(self, idx) -> CR:
        return self.rxns[idx]