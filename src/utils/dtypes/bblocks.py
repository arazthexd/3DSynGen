from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Any

import random
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

class BuildingBlock(Chem.Mol):
    def __init__(self, rdmol: Chem.Mol, idx: int = None, clean: bool = True):
        if clean:
            rdmol = self.clean_mol(rdmol)
        
        super().__init__(rdmol)
        self.idx = idx
    
    @staticmethod
    def clean_mol(mol: Chem.Mol, **kwargs) -> Chem.Mol:
        mol = Chem.Mol(mol)
        mol = rdMolStandardize.ChargeParent(mol)
        if not kwargs.get("removeHs", True):
            mol = Chem.AddHs(mol, addCoords=True)
        return mol

    @classmethod
    def read_from_sdf(cls, sdf_path: str, start_idx: int = 0, 
                      *args, **kwargs) -> List[BuildingBlock]:
        suppl = Chem.SDMolSupplier(sdf_path, *args, **kwargs)
        bbs = [BuildingBlock(mol, idx=i+start_idx) 
               for i, mol in enumerate(tqdm(list(suppl)))]
        return bbs

# @dataclass
# class BuildingBlock:
#     idx: int
#     rdmol: Chem.Mol
#     matched_reacts: List[Tuple[Any, int]] = field(default_factory=list)

#     @staticmethod
#     def clean_mol(mol: Chem.Mol, **kwargs) -> Chem.Mol:
#         mol = rdMolStandardize.ChargeParent(mol)
#         if not kwargs.get("removeHs", True):
#             mol = Chem.AddHs(mol, addCoords=True)
#         return mol

#     @classmethod
#     def read_from_sdf(self, sdf_path: str, start_idx: int = 0, *args, **kwargs):
#         suppl = Chem.SDMolSupplier(sdf_path, *args, **kwargs)
#         bbs = [BuildingBlock(i+start_idx, self.clean_mol(mol, **kwargs))
#                for i, mol in enumerate(tqdm(list(suppl)))]
#         return bbs
    
#     def sample_rr(self):
#         assert len(self.matched_reacts) > 0
#         return random.choice(self.matched_reacts)