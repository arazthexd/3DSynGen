import os
from typing import List

from torch.utils.data import DataLoader
import lightning as L

from ..utils.dtypes import BuildingBlock as BB, CustomReaction as CR
from .dataset import CRBBDataset, CRBBOutput

def make_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)
    
class CRBBDataModule(L.LightningDataModule):
    def __init__(self, 
                 root: str = 'crbb', 
                 batch_size=32,
                 rxns: List[CR] = list(),
                 rxn_files: List[str] = list(),
                 reactant_bbs: List[BB] = list(),
                 reactant_sdfs: List[str] = list(),
                 train_ligs: List[BB] = list(),
                 val_ligs: List[BB] = list(),
                 test_ligs: List[BB] = list(),
                 train_ligsdfs: List[str] = list(),
                 val_ligsdfs: List[str] = list(),
                 test_ligsdfs: List[str] = list(),
                 force_reload: bool = False):
        
        super().__init__()
        self.root = root
        self.train_root = os.path.join(root, 'train')
        self.val_root = os.path.join(root, 'val')
        self.test_root = os.path.join(root, 'test')
        [make_dir(r) for r in [self.root, self.train_root, 
                               self.val_root, self.test_root]]
        
        self.batch_size = batch_size
        self.force_reload = force_reload

        self.rxns = rxns if rxns is not None else []
        self.rxn_files = rxn_files

        self.reactant_bbs = reactant_bbs
        self.reactant_sdfs = reactant_sdfs

        self.train_ligs = train_ligs
        self.train_ligsdfs = train_ligsdfs

        self.val_ligs = val_ligs
        self.val_ligsdfs = val_ligsdfs

        self.test_ligs = test_ligs
        self.test_ligsdfs = test_ligsdfs

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CRBBDataset(root=self.train_root,
                                             rxns=self.rxns,
                                             rxn_files=self.rxn_files,
                                             bbs=self.reactant_bbs,
                                             bb_files=self.reactant_sdfs,
                                             ligs=self.train_ligs,
                                             lig_files=self.train_ligsdfs,
                                             force_reload=self.force_reload)
            self.val_dataset = CRBBDataset(root=self.val_root,
                                            rxns=self.rxns,
                                            rxn_files=self.rxn_files,
                                            bbs=self.reactant_bbs,
                                            bb_files=self.reactant_sdfs,
                                            ligs=self.val_ligs,
                                            lig_files=self.val_ligsdfs,
                                            force_reload=self.force_reload)
        
        if stage == 'test' or stage is None:
            self.test_dataset = CRBBDataset(root=self.test_root,
                                            rxns=self.rxns,
                                            rxn_files=self.rxn_files,
                                            bbs=self.reactant_bbs,
                                            bb_files=self.reactant_sdfs,
                                            ligs=self.test_ligs,
                                            lig_files=self.test_ligsdfs,
                                            force_reload=self.force_reload)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, collate_fn=CRBBOutput.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, collate_fn=CRBBOutput.collate_fn)