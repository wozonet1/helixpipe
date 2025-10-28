from dataclasses import dataclass, field
from typing import List


@dataclass
class DatasetCollectionConfig:
    auxiliary_datasets: List[str] = field(default_factory=list)
    name: str = "no_aux"
