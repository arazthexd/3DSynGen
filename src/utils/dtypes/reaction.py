from typing import List, Tuple, Dict

from rdkit import Chem
from rdkit.Chem import rdChemReactions

class CustomReaction(rdChemReactions.ChemicalReaction):
    max_reactant_map_count = 20

    def __init__(self, smarts: str, idx: int | None = None):
        try:
            rdreaction = rdChemReactions.ReactionFromSmarts(smarts)
        except:
            rdreaction = rdChemReactions.ChemicalReaction(smarts)
        super().__init__(rdreaction)
        assert self.GetNumProductTemplates() == 1

        self.n_reactants = self.GetNumReactantTemplates()
        self.standardize_map_nums()
        self.idx = idx

        self.Initialize()

    @classmethod
    def parse_smarts(self, smarts: str):
        smarts = smarts.split()[0]
        reactants_smarts = smarts.split(">")[0]
        products_list = smarts.split(">")[-1].split(".")
        new_smarts = [reactants_smarts + ">>" + product 
                      for product in products_list]
        return [CustomReaction(nsmarts) for nsmarts in new_smarts]
    
    @classmethod
    def parse_mols(self, reactants: List[Chem.Mol], products: List[Chem.Mol]):
        rxn_smarts = [
            ".".join(
                [Chem.MolToSmarts(r) for r in reactants]
            )+">>"+Chem.MolToSmarts(p) for p in products
        ]
        return [CustomReaction(smarts) for smarts in rxn_smarts]
    
    @classmethod
    def parse_txt(self, txt_path: str, start_idx: int = 0):
        reactions: List[CustomReaction] = []
        with open(txt_path, "r") as f:
            for rxn_smarts in f.readlines():
                if rxn_smarts[0] in ["%", " "] or rxn_smarts == "\n":
                    continue
                reactions.extend(CustomReaction.parse_smarts(rxn_smarts))
        
        [
            reaction.set_reaction_idx(i+start_idx) 
            for i, reaction in enumerate(reactions)
        ]
        return reactions
    
    def set_reaction_idx(self, idx):
        self.idx = idx

    def standardize_map_nums(self) -> None:
        changed_mapnos = dict() 
        for i, rtemplate in enumerate(self.GetReactants()):
            next_map_idx = i * self.max_reactant_map_count + 1
            for atom in rtemplate.GetAtoms():
                atom: Chem.Atom
                mapno = atom.GetPropsAsDict().get("molAtomMapNumber")
                if mapno is not None:
                    atom.SetAtomMapNum(next_map_idx)
                    changed_mapnos[mapno] = next_map_idx
                    next_map_idx += 1
        
        for atom in self.GetProductTemplate(0).GetAtoms():
            atom: Chem.Atom
            mapno = atom.GetPropsAsDict().get("molAtomMapNumber")
            if mapno is not None:
                atom.SetAtomMapNum(changed_mapnos[mapno])
        
    def which_reactant(self, mol: Chem.Mol) -> List[int]:
        # TODO: Extra conditions for reactions...
        reactants = self.GetReactants()
        match_rs = [i for i, reactant in enumerate(reactants) 
                    if mol.HasSubstructMatch(reactant)]
        return match_rs
    
    def is_reactant(self, mol: Chem.Mol) -> List[int]: 
        return self.which_reactant(mol) # backward compatibility! :D

