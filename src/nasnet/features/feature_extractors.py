# 文件: src/nasnet/features/feature_extractors.py (重构后)

from typing import Any, Dict, List

import torch
from transformers import AutoModel, AutoTokenizer, EsmModel

from .base_extractor import BaseFeatureExtractor  # 导入我们的新基类


# --- ESM 蛋白质特征提取器 ---
class EsmFeatureExtractor(BaseFeatureExtractor):
    def _load_model_and_tokenizer(self) -> tuple:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = EsmModel.from_pretrained(self.model_name)
        return model, tokenizer

    def _prepare_batch_input(self, tokenizer: Any, batch_data: List[str]) -> Any:
        return tokenizer(
            batch_data, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

    def _run_model_inference(self, model: Any, inputs: Any) -> Any:
        return model(**inputs)

    def _postprocess_batch_output(
        self, outputs: Any, inputs: Any
    ) -> List[torch.Tensor]:
        token_representations = outputs.last_hidden_state
        embeddings = []
        for i in range(token_representations.size(0)):
            attention_mask = inputs["attention_mask"][i]
            true_token_reprs = token_representations[i][attention_mask == 1][1:-1]
            sequence_level_repr = true_token_reprs.mean(dim=0)
            embeddings.append(sequence_level_repr)
        return embeddings


# --- ChemBERTa 分子特征提取器 ---
class ChembertaFeatureExtractor(BaseFeatureExtractor):
    def _load_model_and_tokenizer(self) -> tuple:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        return model, tokenizer

    def _prepare_batch_input(self, tokenizer: Any, batch_data: List[str]) -> Any:
        return tokenizer(
            batch_data,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        ).to(self.device)

    def _run_model_inference(self, model: Any, inputs: Any) -> Any:
        return model(**inputs)

    def _postprocess_batch_output(
        self, outputs: Any, inputs: Any
    ) -> List[torch.Tensor]:
        # ChemBERTa 直接使用 [CLS] token 的输出
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return [emb for emb in cls_embeddings]  # 转换为列表


# --- 高阶分派函数 ---
def extract_features(
    entity_type: str, config, device, **kwargs
) -> Dict[Any, torch.Tensor]:
    """一个高阶函数，根据 entity_type 动态选择并运行正确的提取器。"""
    if entity_type == "protein":
        extractor = EsmFeatureExtractor(entity_type, config, device)
        return extractor.extract(**kwargs)
    elif entity_type == "molecule":
        extractor = ChembertaFeatureExtractor(entity_type, config, device)
        return extractor.extract(**kwargs)
    else:
        raise ValueError(f"Unknown entity type for feature extraction: {entity_type}")
