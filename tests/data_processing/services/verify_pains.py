# verify_pains.py
from rdkit import Chem
from rdkit.Chem import FilterCatalog

# 创建与我们代码中完全相同的PAINS过滤器
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)  # type: ignore
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)  # type: ignore
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)  # type: ignore
pains_catalog = FilterCatalog.FilterCatalog(params)

# 我们案件中的“嫌疑人”
smiles_to_check = "c1(N=O)cc(O)ccc1"
mol = Chem.MolFromSmiles(smiles_to_check)

# “审问”嫌疑人
is_a_pains_molecule = pains_catalog.HasMatch(mol)

print(f"SMILES: {smiles_to_check}")
print(
    f"Does RDKit's PAINS filter consider this a PAINS molecule? -> {is_a_pains_molecule}"
)

# 让我们看看它匹配到了哪个具体的条目
if is_a_pains_molecule:
    entry = pains_catalog.GetFirstMatch(mol)
    print(f"Matched PAINS filter entry: {entry.GetDescription()}")
else:
    print("No PAINS patterns were matched.")
