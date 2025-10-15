# 文件: src/configs/mlflow.py

from dataclasses import dataclass


@dataclass
class MlflowConfig:
    tracking_uri: str = "logs/mlruns"
    experiment_name: str = (
        "Het-DTI_Prediction_Hydra_on_${data_structure.primary_dataset}"
    )
