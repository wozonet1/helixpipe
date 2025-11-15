# 文件: src/helixpipe/features/base_extractor.py (全新)

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from tqdm import tqdm

import helixlib as hx
from helixpipe.typing import AppConfig, AuthID
from helixpipe.utils import get_path

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """
    一个通用的、基于“每个实体一个缓存文件”策略的特征提取器框架 (模板方法模式)。
    """

    def __init__(self, entity_type: str, config: AppConfig, device: str) -> None:
        self.entity_type = entity_type
        self.config = config
        self.device = device

        # 1. 读取通用配置
        try:
            entity_cfg = self.config.data_params.feature_extractors[entity_type]
            self.model_name = entity_cfg.model_name
            self.batch_size = entity_cfg.batch_size
        except KeyError:
            raise ValueError(
                f"No feature extractor configuration found for entity type '{entity_type}'."
            )

        # 2. 准备缓存路径工厂
        safe_model_name = self.model_name.replace("/", "_")
        self._cache_path_factory = get_path(
            config,
            "cache.features.template",
            entity_type=f"{entity_type}s",  # e.g., 'proteins', 'molecules'
            model_name=safe_model_name,
        )

    # --- 模板方法 (不可修改的骨架) ---
    def extract(
        self,
        authoritative_ids: list[AuthID],
        sequences_or_smiles: list[str],
        force_regenerate: bool = False,
    ) -> dict[Any, torch.Tensor]:
        logger.info(
            f"\n--> [Generic Extractor] Processing {len(authoritative_ids)} {self.entity_type}s using model '{self.model_name}'..."
        )

        results_dict: dict[Any, torch.Tensor] = {}
        missed_ids = []
        missed_data = []

        # --- 1. 缓存命中阶段 ---
        if not force_regenerate:
            for item_id, data_item in zip(authoritative_ids, sequences_or_smiles):
                cache_path = self._cache_path_factory(authoritative_id=item_id)
                if cache_path.exists():
                    results_dict[item_id] = torch.load(cache_path, map_location="cpu")
                else:
                    missed_ids.append(item_id)
                    missed_data.append(data_item)
        else:
            missed_ids = authoritative_ids
            missed_data = sequences_or_smiles

        logger.info(f"    - Cache hits: {len(results_dict)} / {len(authoritative_ids)}")

        # --- 2. 缓存未命中阶段 ---
        if missed_ids:
            logger.info(
                f"    - Cache misses: {len(missed_ids)}. Starting online extraction..."
            )

            # a. 【策略点1】加载模型
            model, tokenizer = self._load_model_and_tokenizer()
            model.to(self.device)
            model.eval()

            progress_bar = tqdm(
                range(0, len(missed_data), self.batch_size),
                desc=f"      {self.entity_type.capitalize()} Batches",
            )

            for i in progress_bar:
                batch_ids = missed_ids[i : i + self.batch_size]
                batch_data = missed_data[i : i + self.batch_size]

                with torch.no_grad():
                    # b. 【策略点2】准备输入
                    inputs = self._prepare_batch_input(tokenizer, batch_data)
                    # c. 【策略点3】模型推理
                    outputs = self._run_model_inference(model, inputs)
                    # d. 【策略点4】后处理
                    embeddings = self._postprocess_batch_output(outputs, inputs)

                # --- 3. 缓存回填阶段 ---
                for j, item_id in enumerate(batch_ids):
                    embedding = embeddings[j].cpu()
                    results_dict[item_id] = embedding

                    cache_path = self._cache_path_factory(authoritative_id=item_id)
                    hx.ensure_path_exists(cache_path)
                    torch.save(embedding, cache_path)

        logger.info(
            f"--> [{self.entity_type.capitalize()} Extractor] Processing complete."
        )
        return results_dict

    # --- 抽象方法 (子类必须实现的“策略”) ---
    @abstractmethod
    def _load_model_and_tokenizer(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def _prepare_batch_input(self, tokenizer: Any, batch_data: list[str]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _run_model_inference(self, model: Any, inputs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _postprocess_batch_output(
        self, outputs: Any, inputs: Any
    ) -> list[torch.Tensor]:
        raise NotImplementedError
