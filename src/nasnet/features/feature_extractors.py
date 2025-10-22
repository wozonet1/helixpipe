from pathlib import Path
from typing import Dict, List

import research_template as rt
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, EsmModel

from nasnet.utils import get_path

# ------------------- 模型加载与初始化 -------------------


# TODO: 使用esm2-3B
def _get_esm_model_and_alphabet(
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    device: str = "cpu",
) -> tuple:
    """
    【新版】从Hugging Face Hub加载指定的ESM-2模型和对应的分词器。
    """
    print(
        f"--> [ESM Extractor] Loading ESM model via Hugging Face: {model_name} onto device: {device}"
    )

    try:
        # 1. 加载分词器 (Tokenizer)，它等同于旧版的 alphabet
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 2. 加载模型本身
        model = EsmModel.from_pretrained(model_name)
    except Exception as e:
        print(
            f"❌ FATAL: Failed to load ESM model from Hugging Face Hub '{model_name}'."
        )
        print(
            "   Please check your internet connection and if the model name is correct on Hugging Face."
        )
        print(f"   Original error: {e}")
        raise

    model.eval()

    try:
        model = model.to(device)
    except Exception as e:
        print(f"❌ FATAL: Failed to move ESM model to device '{device}'.")
        print(
            "   Please check if the specified GPU is available and has enough memory."
        )
        print(f"   Original error: {e}")
        raise

    print(
        f"--> [ESM Extractor] Model '{model_name}' loaded successfully via Hugging Face."
    )
    # 返回 tokenizer 和 model，注意顺序
    return tokenizer, model


# ------------------- 嵌入提取核心逻辑 -------------------


def extract_esm_protein_embeddings(
    authoritative_ids: List[str],
    sequences: List[str],
    config,
    device: str,
    force_regenerate: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    【V2 全局缓存版】智能地提取或加载蛋白质嵌入。

    为给定的UniProt ID列表获取特征。它会优先从全局缓存加载，
    只为缓存中不存在的ID进行在线计算，并将新结果回填到缓存中。

    Args:
        authoritative_ids (List[str]): 有序的UniProt ID列表。
        sequences (List[str]): 与ID列表一一对应的蛋白质序列列表。
        config (DictConfig): 全局配置对象。
        device (str): PyTorch设备。
        force_regenerate (bool): 是否强制重新计算所有特征。

    Returns:
        Dict[str, torch.Tensor]: 一个从UniProt ID映射到其特征张量的字典。
    """
    entity_cfg = config.data_params.feature_extractors.protein
    model_name = entity_cfg.model_name
    batch_size = entity_cfg.batch_size

    print(
        f"\n--> [ESM Extractor] Processing {len(authoritative_ids)} proteins with global cache..."
    )

    results_dict: Dict[str, torch.Tensor] = {}
    missed_ids = []
    missed_sequences = []
    safe_model_name = model_name.replace("/", "_")
    cache_path_factory = get_path(
        config,
        "cache.feature_template",
        entity_type="proteins",
        model_name=safe_model_name,
    )
    # --- 1. 缓存命中阶段 ---
    if not force_regenerate:
        for pid, seq in zip(authoritative_ids, sequences):
            # 构造全局缓存路径
            cache_path = cache_path_factory(authoritative_id=pid)

            if cache_path.exists():
                results_dict[pid] = torch.load(cache_path, map_location="cpu")
            else:
                missed_ids.append(pid)
                missed_sequences.append(seq)
    else:
        # 如果强制重新生成，则所有ID都视为未命中
        missed_ids = authoritative_ids
        missed_sequences = sequences

    print(f"    - Cache hits: {len(results_dict)} / {len(authoritative_ids)}")

    # --- 2. 缓存未命中阶段 ---
    if missed_ids:
        print(f"    - Cache misses: {len(missed_ids)}. Starting online extraction...")
        tokenizer, model = _get_esm_model_and_alphabet(model_name, device)

        progress_bar = tqdm(
            range(0, len(missed_sequences), batch_size), desc="      ESM Batches"
        )

        for i in progress_bar:
            batch_ids = missed_ids[i : i + batch_size]
            batch_seqs = missed_sequences[i : i + batch_size]

            inputs = tokenizer(
                batch_seqs, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                token_representations = outputs.last_hidden_state

            # --- 3. 缓存回填阶段 (在循环内部) ---
            for j, pid in enumerate(batch_ids):
                attention_mask = inputs["attention_mask"][j]
                true_token_reprs = token_representations[j][attention_mask == 1][1:-1]
                sequence_level_repr = true_token_reprs.mean(dim=0).cpu()

                # a. 更新结果字典
                results_dict[pid] = sequence_level_repr

                # b. 保存到全局缓存
                cache_path = cache_path_factory(authoritative_id=pid)
                rt.ensure_path_exists(cache_path)
                torch.save(sequence_level_repr, cache_path)

    print("--> [ESM Extractor] Processing complete.")
    return results_dict


def _get_chemberta_model_and_tokenizer(
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1", device: str = "cpu"
) -> tuple:
    """
    从Hugging Face Hub加载指定的ChemBERTa模型和对应的分词器。
    """
    print(
        f"--> [ChemBERTa Extractor] Loading model via Hugging Face: {model_name} onto device: {device}"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    except Exception as e:
        print(f"❌ FATAL: Failed to load ChemBERTa model '{model_name}'.")
        print(f"   Original error: {e}")
        raise

    model.eval()
    model = model.to(device)

    print(f"--> [ChemBERTa Extractor] Model '{model_name}' loaded successfully.")
    return tokenizer, model


# ------------------- 嵌入提取核心逻辑 -------------------


# TODO: 添加lru装饰器
def extract_chemberta_molecule_embeddings(
    authoritative_ids: List[int],  # <-- 签名变化: 接收 CID 列表
    smiles_list: List[str],  # <-- 签名变化: 与ID一一对应
    config,
    device: str,
    force_regenerate: bool = False,
) -> Dict[int, torch.Tensor]:  # <-- 返回值变为字典
    """
    【V2 全局缓存版】智能地提取或加载小分子（SMILES）嵌入。

    为给定的PubChem CID列表获取特征。优先从全局缓存加载，
    只为缓存中不存在的ID进行在线计算，并将新结果回填。

    Args:
        authoritative_ids (List[int]): 有序的PubChem CID列表。
        smiles_list (List[str]): 与ID列表一一对应的SMILES字符串列表。
        config (DictConfig): 全局配置对象。
        device (str): PyTorch设备。
        force_regenerate (bool): 是否强制重新计算所有特征。

    Returns:
        Dict[int, torch.Tensor]: 一个从PubChem CID映射到其特征张量的字典。
    """
    entity_cfg = config.data_params.feature_extractors.molecule
    model_name = entity_cfg.model_name
    batch_size = entity_cfg.batch_size

    print(
        f"\n--> [ChemBERTa Extractor] Processing {len(authoritative_ids)} molecules with global cache..."
    )

    results_dict: Dict[int, torch.Tensor] = {}
    missed_ids = []
    missed_smiles = []
    safe_model_name = model_name.replace("/", "_")
    cache_path_factory = get_path(
        config,
        "cache.feature_template",
        entity_type="molecules",
        model_name=safe_model_name,
    )
    # --- 1. 缓存命中阶段 ---
    if not force_regenerate:
        for cid, smiles in zip(authoritative_ids, smiles_list):
            # model_name中包含'/'，需要替换为'_'以作为合法的文件夹名

            cache_path: Path = cache_path_factory(authoritative_id=cid)

            if cache_path.exists():
                results_dict[cid] = torch.load(cache_path, map_location="cpu")
            else:
                missed_ids.append(cid)
                missed_smiles.append(smiles)
    else:
        missed_ids = authoritative_ids
        missed_smiles = smiles_list

    print(f"    - Cache hits: {len(results_dict)} / {len(authoritative_ids)}")

    # --- 2. 缓存未命中阶段 ---
    if missed_ids:
        print(f"    - Cache misses: {len(missed_ids)}. Starting online extraction...")
        tokenizer, model = _get_chemberta_model_and_tokenizer(model_name, device)

        progress_bar = tqdm(
            range(0, len(missed_smiles), batch_size), desc="      ChemBERTa Batches"
        )

        for i in progress_bar:
            batch_ids = missed_ids[i : i + batch_size]
            batch_smiles = missed_smiles[i : i + batch_size]

            inputs = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # 使用 [CLS] token 的输出作为分子表示
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

            # --- 3. 缓存回填阶段 (在循环内部) ---
            for j, cid in enumerate(batch_ids):
                embedding = batch_embeddings[j]

                # a. 更新结果字典
                results_dict[cid] = embedding

                # b. 保存到全局缓存
                cache_path = cache_path_factory(authoritative_id=cid)
                rt.ensure_path_exists(cache_path)
                torch.save(embedding, cache_path)

    print("--> [ChemBERTa Extractor] Processing complete.")
    return results_dict
