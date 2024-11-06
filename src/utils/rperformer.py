import os
from typing import List, Tuple, Dict, Optional, Any
from tqdm import tqdm
import dill
import random
import warnings
warnings.simplefilter("ignore")

from rdkit import Chem
from rdkit.Chem import (
    rdDistGeom, rdMolAlign, rdChemReactions, rdForceFieldHelpers, 
)
from rdkit.Geometry import rdGeometry
from rdkit import rdBase
rdBase.DisableLog("rdApp.*")

class Reaction3DPerformer:
    max_reactant_map_count = 20

    @staticmethod
    def prepare_reactant(mol: Chem.Mol, reactant_id: int, embed: bool = True,
                         sample_confs: bool = False) -> Chem.Mol:
        mol = Chem.Mol(mol)

        if sample_confs:
            n_conf = mol.GetNumConformers()
            rand_conf = mol.GetConformers()[random.randint(0, n_conf)]
            mol.RemoveAllConformers()
            mol.AddConformer(rand_conf, assignId=True)
            mol = Chem.AddHs(mol, addCoords=True)
        elif embed:
            mol = Chem.AddHs(mol)
            rdDistGeom.EmbedMolecule(mol)

        Chem.AssignStereochemistryFrom3D(mol)
        mol = Chem.RemoveHs(mol, updateExplicitCount=True) # TODO: Decide whether this should exist
        [a.SetIntProp("react_mol_id", reactant_id) for a in mol.GetAtoms()]
        return mol
    
    @classmethod
    def prepare_reactants(cls, reactants: List[Chem.Mol], embed: bool = True,
                          sample_confs: bool = False) -> List[Chem.Mol]:
        return [cls.prepare_reactant(mol, i, embed, sample_confs) 
                for i, mol in enumerate(reactants)]
    
    @staticmethod
    def prepare_products(products: List[Chem.Mol]):
        try:
            [p.UpdatePropertyCache() for p in products]
        except:
            [a.SetNumExplicitHs(0) for p in products for a in p.GetAtoms()]
            [p.UpdatePropertyCache() for p in products] # TODO: !!!
        [Chem.SanitizeMol(p) for p in products]
        products = [Chem.AddHs(p) for p in products]
        [rdDistGeom.EmbedMolecule(p, maxAttempts=1) for p in products]
        [rdForceFieldHelpers.MMFFOptimizeMolecule(
            p, maxIters=15, nonBondedThresh=3) 
            for p in products if p.GetNumConformers() > 0]
        return products
    
    @classmethod
    def match_reactants_to_product(cls, product: Chem.Mol, n_reactants: int):
        match_atom_ids_direct = [dict() for _ in range(n_reactants)]
        match_atom_ids_indirect = [dict() for _ in range(n_reactants)]
        match_atom_coords = [dict() for _ in range(n_reactants)]
        for patom in product.GetAtoms():
            patom: Chem.Atom
            react_atom_idx = patom.GetPropsAsDict().get("react_atom_idx")
            reactant_id = patom.GetPropsAsDict().get("react_mol_id")
            mapno = patom.GetPropsAsDict().get("old_mapno")

            if mapno is not None and react_atom_idx is not None:
                reactant_id = mapno // cls.max_reactant_map_count
                match_atom_ids_indirect[reactant_id][react_atom_idx] = \
                    patom.GetIdx()
                match_atom_coords[reactant_id][react_atom_idx] = \
                    product.GetConformer().GetAtomPosition(patom.GetIdx())
                
            elif reactant_id is not None and react_atom_idx is not None:
                match_atom_ids_direct[reactant_id][react_atom_idx] = patom.GetIdx()  
                match_atom_coords[reactant_id][react_atom_idx] = \
                    product.GetConformer().GetAtomPosition(patom.GetIdx())
                
        return list(zip(match_atom_ids_direct, 
                        match_atom_ids_indirect, 
                        match_atom_coords))
    
    @classmethod
    def perform_reaction(cls, reaction: rdChemReactions.ChemicalReaction, 
                         reactants: List[Chem.Mol], embed_rs: bool = False):
        # Primary check
        for i, reactant in enumerate(reactants):
            assert reactant.HasSubstructMatch(reaction.GetReactantTemplate(i))

        # Prepare reactants
        reactants = cls.prepare_reactants(reactants, embed=embed_rs)
        
        # Perform, sanitize products, return None if failed in any part.
        possible_products = [p[0] for p in reaction.RunReactants(reactants)]
        possible_products: List[Chem.Mol]
        possible_products = cls.prepare_products(possible_products)
        if all([p.GetNumConformers() == 0 for p in possible_products]):
            return None
        else:
            possible_products = [p for p in possible_products 
                                 if p.GetNumConformers() > 0]
    
        # Find reactant atom matches to products and their coordinates.
        matches: List[List[Tuple[List[int], List[rdGeometry.Point3D]]]]
        matches = [cls.match_reactants_to_product(product, len(reactants)) 
                   for product in possible_products]
        
        # Create new reactant atom coords and align with product.
        out = []
        for product, rmatches in zip(possible_products, matches):
            out_reactants = []
            for reactant, match in zip(reactants, rmatches):
                if match is None:
                    continue
                reactant = Chem.AddHs(reactant)
                success = rdDistGeom.EmbedMolecule(reactant, coordMap=match[2])
                if success == -1:
                    # print("Embedding failed")
                    return None
                rmsd_orig = rdMolAlign.AlignMol(
                    reactant, product, 
                    atomMap=list({**match[0],**match[1]}.items())
                )
                r_orig = Chem.Mol(reactant)
                rmsd_refl = rdMolAlign.AlignMol(
                    reactant, product, reflect=True,
                    atomMap=list({**match[0],**match[1]}.items())
                )
                if rmsd_orig < rmsd_refl:
                    reactant = r_orig
                out_reactants.append(reactant)
            out.append((out_reactants, product))
        
        return out