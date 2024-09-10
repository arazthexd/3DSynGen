from typing import List, Tuple

import random

from rdkit import Chem
from rdkit.Chem import (
    rdChemReactions, rdDistGeom, rdGeometry, rdMolAlign, rdForceFieldHelpers
)

from .bblocks import BuildingBlock as BB

class Reaction3D:
    max_reactant_map_count = 20
    def __init__(self, smarts: str, idx: int | None = None):
        self.rxn = rdChemReactions.ReactionFromSmarts(smarts)
        self.n_reactants = self.rxn.GetNumReactantTemplates()
        assert self.rxn.GetNumProductTemplates() == 1
        self.possible_reactants: List[List[Chem.Mol]] = [
            list() for _ in range(self.n_reactants)
        ]
        self.standardize_map_nums(self.rxn.GetReactants(), 
                                  self.rxn.GetProductTemplate(0))
        
        self.idx = idx

    @classmethod
    def parse_smarts(self, smarts: str):
        reactants_smarts = smarts.split(">>")[0]
        products_list = smarts.split(">>")[1].split(".")
        new_smarts = [reactants_smarts + ">>" + product 
                      for product in products_list]
        return [Reaction3D(nsmarts) for nsmarts in new_smarts]
    
    @classmethod
    def parse_mols(self, reactants: List[Chem.Mol], products: List[Chem.Mol]):
        rxn_smarts = [
            ".".join(
                [Chem.MolToSmarts(r) for r in reactants]
            )+">>"+Chem.MolToSmarts(p) for p in products
        ]
        return [Reaction3D(smarts) for smarts in rxn_smarts]
    
    @classmethod
    def parse_txt(self, txt_path: str, start_idx: int = 0):
        reactions: List[Reaction3D] = []
        with open(txt_path, "r") as f:
            for rxn_smarts in f.readlines():
                reactions.extend(Reaction3D.parse_smarts(rxn_smarts))
        
        [
            reaction.set_reaction_idx(i+start_idx) 
            for i, reaction in enumerate(reactions)
        ]
        return reactions
    
    def set_reaction_idx(self, idx: int):
        self.idx = idx

    def standardize_map_nums(self, reactants: List[Chem.Mol], 
                             product: Chem.Mol) -> None:
        changed_mapnos = dict() 
        for i, rtemplate in enumerate(self.rxn.GetReactants()):
            next_map_idx = i * self.max_reactant_map_count + 1
            for atom in rtemplate.GetAtoms():
                atom: Chem.Atom
                mapno = atom.GetPropsAsDict().get("molAtomMapNumber")
                if mapno is not None:
                    atom.SetAtomMapNum(next_map_idx)
                    changed_mapnos[mapno] = next_map_idx
                    next_map_idx += 1
        
        for atom in product.GetAtoms():
            atom: Chem.Atom
            mapno = atom.GetPropsAsDict().get("molAtomMapNumber")
            if mapno is not None:
                atom.SetAtomMapNum(changed_mapnos[mapno])
        
    def is_reactant(self, mol: Chem.Mol) -> List[int]:
        # TODO: Extra conditions for reactions...
        reactants = self.rxn.GetReactants()
        match_rs = [i for i, reactant in enumerate(reactants) if mol.HasSubstructMatch(reactant)]
        return match_rs

    def add_potential_mols(self, mols: List[Chem.Mol]) -> None:
        [self.possible_reactants[rid].append(mol) 
         for mol in mols 
         for rid in self.is_reactant(mol)]
        
    def add_potential_bbs(self, bbs: List[BB],
                          add_to_bb: bool = False):
        def add_this(rid: int, bb: BB):
            self.possible_reactants[rid].append(bb) 
            if add_to_bb:
                bb.matched_reacts.append((self.idx, rid))

        [add_this(rid, bb)
         for bb in bbs 
         for rid in self.is_reactant(bb.rdmol)]

    def reset_potentials(self) -> None:
        self.possible_reactants: List[List[Chem.Mol]] = [
            list() for _ in range(self.n_reactants)
        ]
    
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
        mol = Chem.RemoveHs(mol)
        [a.SetIntProp("react_mol_id", reactant_id) for a in mol.GetAtoms()]
        return mol
    
    def prepare_reactants(self, reactants: List[Chem.Mol], embed: bool = True,
                          sample_confs: bool = False) -> List[Chem.Mol]:
        return [self.prepare_reactant(mol, i, embed, sample_confs) 
                for i, mol in enumerate(reactants)]
    
    def match_reactants_to_product(self, product: Chem.Mol):
        match_atom_ids_direct = [dict() for _ in range(self.n_reactants)]
        match_atom_ids_indirect = [dict() for _ in range(self.n_reactants)]
        match_atom_coords = [dict() for _ in range(self.n_reactants)]
        for patom in product.GetAtoms():
            patom: Chem.Atom
            react_atom_idx = patom.GetPropsAsDict().get("react_atom_idx")
            reactant_id = patom.GetPropsAsDict().get("react_mol_id")
            mapno = patom.GetPropsAsDict().get("old_mapno")

            if mapno is not None and react_atom_idx is not None:
                reactant_id = mapno // self.max_reactant_map_count
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

    def run_reaction(
            self, 
            reactants: List[Chem.Mol]
        ) -> List[Tuple[List[Chem.Mol], Chem.Mol]]:

        # Prepare reactants.
        reactants = self.prepare_reactants(reactants)

        # Get possible 3D products.
        possible_products = [p[0] for p in self.rxn.RunReactants(reactants)]
        possible_products: List[Chem.Mol]
        [p.UpdatePropertyCache() for p in possible_products]
        [Chem.SanitizeMol(p) for p in possible_products]
        # [Chem.FindRingFamilies(p) for p in possible_products]
        possible_products = [Chem.AddHs(p) for p in possible_products]
        [rdDistGeom.EmbedMolecule(p) for p in possible_products]
        [rdForceFieldHelpers.MMFFOptimizeMolecule(
            p, maxIters=30, nonBondedThresh=5) for p in possible_products]

        # Find reactant atom matches to products and their coordinates.
        matches: List[List[Tuple[List[int], List[rdGeometry.Point3D]]]]
        matches = [self.match_reactants_to_product(product) 
                   for product in possible_products]

        # Create new reactant atom coords and align with product.
        out = []
        for product, rmatches in zip(possible_products, matches):
            out_reactants = []
            for reactant, match in zip(reactants, rmatches):
                print(match)
                if match is None:
                    continue
                reactant = Chem.AddHs(reactant)
                success = rdDistGeom.EmbedMolecule(reactant, coordMap=match[2])
                if success == -1:
                    print("Embedding failed")
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

    def sample_reactants(self):
        assert all(len(mpr) > 0 for mpr in self.possible_reactants)
        return [random.choice(mpr) for mpr in self.possible_reactants]

# reaction = Reaction3D("[C,c:1]-[C:2](=[O:3])-[O;H0&-,H1:4].[N&H2:11]-[C:12]-[C:13]-[O&H1:14]>>[C,c:1]-[C:2]1=[N&H0:11]-[C:12]-[C:13]-[O&H0:14]-1")
# r1 = Chem.MolFromSmiles("c1ccccc1C(=O)O")
# r2 = Chem.MolFromSmiles("OC[C@H](N)C(C)CC")
# print(reaction.run_reaction([r1, r2]))
# print(rdChemReactions.ReactionToSmarts(reaction.rxn))

# reaction.setup_reactants([r1, r2])
# print(reaction.possible_reactants)