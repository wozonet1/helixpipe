from rdkit import Chem


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        print(f"Warning: Invalid SMILES string found and will be ignored: {smiles}")
        return None  # Return None for invalid SMILES
