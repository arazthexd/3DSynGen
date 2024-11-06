from __future__ import annotations
import os, shutil, copy, time
from typing import List, Tuple, Dict, Optional, Any, Type
from tqdm import tqdm
import dill
import random
from functools import partial
from dataclasses import dataclass
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
from torch_geometric.data import Data, Dataset, Batch

from ...utils.dtypes import CustomReaction as CR
from ...utils.dtypes import BuildingBlock as BB
from ...utils.helper import draw_cr, chi, cpsa, get_bond_edges, is_reactant
from ...utils.rperformer import Reaction3DPerformer
from .bbdataset import BBDataset, BBFeaturizer, BBConfFeaturizer
from .crdataset import CRDataset, CRFeaturizer

import time

@dataclass
class CRBBOutput:
    lig0_feats: Tuple[Data, Data] # 2d, 3d
    rsfeats_2d: List[Data]
    rsfeats_3d: List[Data]
    rslengths: int | torch.Tensor
    terminate: torch.Tensor
    pfeats: Tuple[Data, Data]
    rxnfeats: torch.Tensor

    @classmethod
    def from_crbb(cls, rsfeats, rscfeats, pfeats, pcfeats, lig0_rid, rxnfeats,
                  maxseqlen):
        rscfeats = [f[0] for f in rscfeats]

        rsfeats, seqlen = cls.pad_rsfeats(rsfeats, maxseqlen)
        rscfeats, _ = cls.pad_rsfeats(rscfeats, maxseqlen)
        lig0feats = rsfeats[lig0_rid]
        lig0cfeats = rscfeats[lig0_rid]

        terminate = torch.zeros(len(rsfeats), dtype=torch.bool)
        terminate[lig0_rid] = True

        return cls(
            lig0_feats=(lig0feats, lig0cfeats),
            rsfeats_2d=rsfeats,
            rsfeats_3d=rscfeats,
            rslengths=seqlen,
            terminate=terminate,
            pfeats=(pfeats, pcfeats[0]),
            rxnfeats=rxnfeats
        )
    
    @staticmethod
    def collate_fn(batch: List[CRBBOutput]):
        lig0feats = Batch.from_data_list([b.lig0_feats[0] for b in batch])
        lig0cfeats = Batch.from_data_list([b.lig0_feats[1] for b in batch])

        rsfeats_2d = [Batch.from_data_list([b.rsfeats_2d[i] for b in batch]) 
                      for i in range(len(batch[0].rsfeats_2d))]
        rsfeats_3d = [Batch.from_data_list([b.rsfeats_3d[i] for b in batch])
                      for i in range(len(batch[0].rsfeats_3d))]
        rslengths = torch.tensor([b.rslengths for b in batch], 
                                 dtype=torch.long)
        terminate = torch.stack([b.terminate for b in batch])

        pfeats = Batch.from_data_list([b.pfeats[0] for b in batch])
        pcfeats = Batch.from_data_list([b.pfeats[1] for b in batch])
        
        rxnfeats = torch.stack([b.rxnfeats for b in batch])
        
        return CRBBOutput(
            lig0_feats=(lig0feats, lig0cfeats),
            rsfeats_2d=rsfeats_2d,
            rsfeats_3d=rsfeats_3d,
            rslengths=rslengths,
            terminate=terminate,
            pfeats=(pfeats, pcfeats),
            rxnfeats=rxnfeats
        )

    @staticmethod
    def pad_rsfeats(rsfeats: List[Data], max_len: int = 5):
        seqlen = len(rsfeats)
        if seqlen <= max_len:
            for _ in range(max_len - seqlen):
                pad_dict = {k: torch.zeros_like(v) 
                            for k, v in rsfeats[0].to_dict().items()}
                pad = Data.from_dict(pad_dict)
                rsfeats += [pad]
        else:
            print("WARNING! MAX LEN OF SEQ IS NOT ENOUGH!")
            rsfeats = rsfeats[:max_len]
        
        return rsfeats, seqlen

class CRBBDataset(Dataset):
    def __init__(self, root: str = ".", 
                 rxns: List[CR] = list(), rxn_files: List[str] = list(),
                 bbs: List[BB] = list(), bb_files: List[str] = list(),
                 rxn_featurizer: CRFeaturizer = CRFeaturizer(), 
                 rxn_feats: torch.Tensor | List[Data] = None,
                 bb_featurizer: BBFeaturizer = BBFeaturizer(), 
                 bb_feats: torch.Tensor | List[Data] = None,
                 bbconf_featurizer: BBConfFeaturizer = BBConfFeaturizer(), 
                 bbconf_feats: torch.Tensor | List[Data] = None,
                 epoch_multiplier: int = 20, 
                 maxseqlen: int = 3,
                 force_reload: bool = False):
        
        self.bbd_root = os.path.join(root, 'bbs')
        self.crd_root = os.path.join(root, 'rxns')

        print("Creating building block dataset...")
        self.bbdataset = BBDataset(root=self.bbd_root,
                                   bbs=bbs, sdf_files=bb_files,
                                   featurizer=bb_featurizer,
                                   feats=bb_feats,
                                   cfeaturizer=bbconf_featurizer,
                                   cfeats=bbconf_feats,
                                   use_confs=True,
                                   force_reload=force_reload)
        print("Creating reaction dataset...")
        self.crdataset = CRDataset(root=self.crd_root,
                                   rxns=rxns, rxn_files=rxn_files,
                                   featurizer=rxn_featurizer,
                                   feats=rxn_feats,
                                   force_reload=force_reload)
        
        super().__init__(root, force_reload=force_reload)

        self.epoch_multiplier = epoch_multiplier
        self.pregen_data = []
        self.is_pregened = False
        self.maxseqlen = maxseqlen

    @property 
    def raw_file_names(self): 
        return []
    
    @property 
    def processed_file_names(self): 
        return ['data_dict.pt']
    
    @property
    def bbs(self) -> List[BB]:
        return self.bbdataset.bbs
    
    @property
    def rxns(self) -> List[CR]:
        return self.crdataset.rxns
    
    @property
    def bbfeats(self) -> torch.Tensor | List[Data]:
        return self.bbdataset.feats
    
    @property
    def bbcfeats(self) -> torch.Tensor | List[Data]:
        return self.bbdataset.cfeats
    
    @property
    def rxnfeats(self) -> torch.Tensor | List[Data]:
        return self.crdataset.feats
    
    @property
    def bbfeaturizer(self) -> BBFeaturizer:
        return self.bbdataset.featurizer
    
    @property
    def bbcfeaturizer(self) -> BBConfFeaturizer:
        return self.bbdataset.cfeaturizer

    @property
    def rxnfeaturizer(self) -> CRFeaturizer:
        return self.crdataset.featurizer

    def download(self):
        pass

    def process(self):

        print("Matching building blocks and reactions...")
        self.rxnid2bbid = {}
        self.rxnids = []
        for rxn_id, rxn in enumerate(tqdm(self.rxns)):
            rxn_bb_matches = [
                [
                    bb_id for bb_id, bb in enumerate(self.bbs) 
                    if rid in rxn.is_reactant(bb)
                ] for rid in range(rxn.n_reactants)
            ]
            
            if any(len(matches) == 0 for matches in rxn_bb_matches):
                continue

            self.rxnid2bbid[rxn_id] = rxn_bb_matches
            self.rxnids.append(rxn_id)
        
        self.bbid2rxnrid = {}
        self.bbids = []
        for bb_id, bb in enumerate(tqdm(self.bbs)):
            bb_rxn_matches = [
                (rxn_id, rid) for rxn_id in self.rxnids
                for rid in self.rxns[rxn_id].is_reactant(bb)
            ]

            if len(bb_rxn_matches) == 0:
                continue

            self.bbid2rxnrid[bb_id] = bb_rxn_matches
            self.bbids.append(bb_id)

        print("Constructing searcher for every rxn reactant...")
        self.rxnid2bbsearchers = {}
        for rxn_id, rid2bbids in tqdm(self.rxnid2bbid.items()):
            rid2bbsearcher = []
            for rid, bbids in enumerate(rid2bbids):
                if isinstance(self.bbfeats[0], torch.Tensor):
                    bbfeats = self.bbfeats[bbids]
                else:
                    bbfeats = [self.bbfeats[i].y for i in bbids]
                    bbfeats = torch.cat(bbfeats, dim=0)
                
                bbsearcher = NearestNeighbors(n_neighbors=5).fit(bbfeats)
                rid2bbsearcher.append(bbsearcher)
            
            self.rxnid2bbsearchers[rxn_id] = rid2bbsearcher
        
        print("Constructing conformer searcher for every building block...")
        self.bbid2bbcsearcher = {}
        for bbid, bb in enumerate(tqdm(self.bbs)):
            bbcfeats = torch.cat([data.y for data in self.bbcfeats[bbid]], dim=0)
            bbcsearcher = NearestNeighbors(n_neighbors=1).fit(bbcfeats)
            self.bbid2bbcsearcher[bbid] = bbcsearcher
                
    def len(self):
        if self.is_pregened:
            return len(self.pregen_data)
        else:
            return len(self.bbid2rxnrid) * self.epoch_multiplier
    
    def get(self, idx):
        if self.is_pregened:
            return self.pregen_data[idx]
        
        else:
            return self.generate(idx)
    
    def generate(self, idx):
        idx = idx % len(self.bbid2rxnrid)

        bb_id = self.bbids[idx]
        rxn_id, rid = self.sample_bbrxn(bb_id)

        rs, p = self.sample_rxnresult(rxn_id, 
                                      l0=self.bbs[bb_id], 
                                      l0_rid=rid)
        
        rsfeats = self.bbfeaturizer.featurize(rs, progbar=False, transform_only=True)
        rscfeats = self.bbcfeaturizer.featurize(rs, progbar=False)
        pfeats = self.bbfeaturizer.featurize([p], progbar=False, transform_only=True)[0]
        pcfeats = self.bbcfeaturizer.featurize([p], progbar=False)[0]

        rxnfeats = self.encode_rxn(rxn_id, rs, p)

        output = CRBBOutput.from_crbb(rsfeats, rscfeats, pfeats, pcfeats, 
                                      rid, rxnfeats, self.maxseqlen)
        return output
    
    def encode_rxn(self, rxn_id: int, rs: List[Chem.Mol], p: Chem.Mol):
        return self.rxnfeats[rxn_id]
    
    def sample_bbrxn(self, bb_id):
        rxn_id, rid = random.choice(self.bbid2rxnrid[bb_id])
        return rxn_id, rid
    
    def sample_rxnbb(self, rxn_id):
        rxn = self.rxns[rxn_id]
        matching_rid2bbid = self.rxnid2bbid[rxn_id]
        bbs = []
        for rid, matching_bbids in enumerate(matching_rid2bbid):
            bb_id = random.choice(matching_bbids)
            bbs.append(self.bbs[bb_id])
        return rxn, bbs
    
    def sample_rxnresult(self, idx, l0: Chem.Mol = None, l0_rid: int = None, 
                         return_all: bool = False):
        rxn, bbs = self.sample_rxnbb(idx)
        
        if l0:
            assert l0_rid is not None
            bbs[l0_rid] = l0
        
        try:
            results = Reaction3DPerformer.perform_reaction(rxn, bbs)
            assert results
        except:
            return self.sample_rxnresult(idx, return_all)
        if return_all:
            return results
        return random.choice(results)
    
    def pregenerate_data(self, multiplier: int = 1, clear: bool = False, 
                         n_proc: int = 1):
        if clear:
            self.pregen_data = []

        print("Pre-generating data for dataset...")
        
        def generate(idx):
            return self.generate(idx)

        for _ in range(multiplier):
            with WorkerPool(n_jobs=n_proc) as pool:
                newdata = pool.map(generate, range(len(self.bbids)), 
                                   progress_bar=True)
            self.pregen_data.extend(newdata)
        self.is_pregened = True

    def save_pregen(self):
        torch.save(self.pregen_data, os.path.join(self.root, "pregen_data.pt"))
    
    def load_pregen(self):
        self.pregen_data = torch.load(os.path.join(self.root, "pregen_data.pt"))
        self.is_pregened = True
