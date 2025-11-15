from dataclasses import dataclass, field


@dataclass
class DatasetCollectionConfig:
    auxiliary_datasets: list[str] = field(default_factory=list)
    name: str = "no_aux"
