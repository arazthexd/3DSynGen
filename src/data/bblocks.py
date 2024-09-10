from dataclasses import dataclass, field
from typing import List, Tuple

import random

from rdkit import Chem

@dataclass
class BuildingBlock:
    idx: int
    rdmol: Chem.Mol
    matched_reacts: List[Tuple[int, int]] = field(default_factory=list)

    @classmethod
    def read_from_sdf(self, sdf_path: str, *args, **kwargs):
        suppl = Chem.SDMolSupplier(sdf_path, *args, **kwargs)
        bbs = [BuildingBlock(i, mol) for i, mol in enumerate(suppl)]
        return bbs
    
    def sample_rr(self):
        assert len(self.matched_reacts) > 0
        return random.choice(self.matched_reacts)