# ==============================================================================
#                 SOTA Protein Feature Extractor (Library Module)
# ==============================================================================
#
# 职责:
#   - 提供使用最先进的ESM-2模型来提取蛋白质序列嵌入的功能。
#   - 设计为被主数据处理流水线 (`main_pipeline.py`) 调用的可缓存模块。
#   - 缓存的读取/写入/强制刷新行为，完全由调用者通过参数控制。
#
# ==============================================================================

import torch
import esm
from tqdm import tqdm
from pathlib import Path
import pickle as pkl
import warnings

# 导入您自己的工具包，用于确保路径存在
from research_template import ensure_path_exists

# ------------------- 模型加载与初始化 -------------------


def get_esm_model_and_alphabet(
    model_name: str = "facebook/esm2_t30_150M_UR50D", device: str = "cpu"
) -> tuple:
    """
    从esm.pretrained加载指定的ESM-2模型和对应的alphabet（分词器）。

    这是一个重量级操作，包含模型下载（首次）和加载到指定设备。

    Args:
        model_name (str): 要加载的ESM模型名称。
        device (str): PyTorch设备字符串，例如 "cuda:0" 或 "cpu"。

    Returns:
        tuple: (model, alphabet)，分别是加载好的ESM模型和用于序列转换的alphabet对象。
    """
    print(f"--> [ESM Extractor] Loading ESM model: {model_name} onto device: {device}")

    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    except Exception as e:
        print(
            f"❌ FATAL: Failed to load ESM model '{model_name}'. Check internet connection and model name."
        )
        print(f"   Original error: {e}")
        raise

    model.eval()  # 关键：设置为评估模式

    try:
        model = model.to(device)
    except Exception as e:
        print(
            f"❌ FATAL: Failed to move ESM model to device '{device}'. Check GPU availability and memory."
        )
        print(f"   Original error: {e}")
        raise

    print(f"--> [ESM Extractor] Model '{model_name}' loaded successfully.")
    return model, alphabet


# ------------------- 嵌入提取核心逻辑 -------------------


def extract_esm_protein_embeddings(
    sequences: list[str],
    cache_path: Path,
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    batch_size: int = 32,
    device: str = "cpu",
    force_regenerate: bool = False,
) -> torch.Tensor:
    """
    【核心接口】智能地提取或加载蛋白质嵌入。

    它首先检查缓存路径。如果缓存文件存在且不强制重新生成，则直接加载。
    否则，它将执行完整的、耗时的在线提取过程，并在完成后保存结果到缓存。

    Args:
        sequences (list[str]): 包含待处理蛋白质氨基酸序列的列表。
        cache_path (Path): 指向缓存文件 (.pkl) 的完整路径对象。
        model_name (str): 要加载的ESM模型名称。
        batch_size (int): 每次送入模型进行推理的序列数量。
        device (str): PyTorch设备字符串。
        force_regenerate (bool): 如果为True，将忽略现有缓存，强制重新生成嵌入。

    Returns:
        torch.Tensor: 一个形状为 [num_sequences, embedding_dim] 的张量。
    """
    # 1. 检查缓存
    if cache_path.exists() and not force_regenerate:
        print(
            f"--> [ESM Extractor] Loading cached protein embeddings from: {cache_path}"
        )
        try:
            with open(cache_path, "rb") as f:
                return pkl.load(f)
        except Exception as e:
            print(
                f"⚠️ WARNING: Failed to load cache file '{cache_path}'. Will regenerate. Error: {e}"
            )

    # 2. 如果缓存不存在或需要强制刷新，则执行在线提取
    if force_regenerate:
        print(
            "--> [ESM Extractor] `force_regenerate` is True. Starting online extraction..."
        )
    else:
        print(
            f"--> [ESM Extractor] No cache found at '{cache_path}'. Starting online extraction..."
        )

    if not sequences:
        warnings.warn("Input sequence list is empty. Returning an empty tensor.")
        # 返回一个0行，但维度正确的空张量
        temp_model, _ = get_esm_model_and_alphabet(model_name, device)
        embedding_dim = temp_model.embed_dim
        del temp_model  # 释放内存
        return torch.empty((0, embedding_dim))

    # a. 加载模型
    model, alphabet = get_esm_model_and_alphabet(model_name, device)

    # b. 批量提取
    all_embeddings = []
    batch_converter = alphabet.get_batch_converter()
    progress_bar = tqdm(
        range(0, len(sequences), batch_size),
        desc="[ESM Extractor] Online Protein Batches",
    )

    for i in progress_bar:
        batch_seqs = sequences[i : i + batch_size]
        data_for_converter = list(
            zip([f"p_{j}" for j in range(len(batch_seqs))], batch_seqs)
        )
        _, _, batch_tokens = batch_converter(data_for_converter)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            try:
                results = model(
                    batch_tokens, repr_layers=[model.num_layers], return_contacts=False
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"\n❌ CUDA Out of Memory on batch size {batch_size}. Try reducing it."
                    )
                else:
                    print(f"\n❌ Unexpected RuntimeError during ESM inference: {e}")
                raise

        token_representations = results["representations"][model.num_layers]

        sequence_representations = []
        for j, seq_string in enumerate(batch_seqs):
            true_token_reprs = token_representations[j, 1 : len(seq_string) + 1]
            sequence_level_repr = true_token_reprs.mean(dim=0)
            sequence_representations.append(sequence_level_repr.cpu())

        all_embeddings.extend(sequence_representations)

    embeddings_tensor = torch.stack(all_embeddings)

    # c. 保存到缓存以备后用
    print(
        f"--> [ESM Extractor] Saving newly generated embeddings to cache: {cache_path}"
    )
    ensure_path_exists(cache_path)  # 确保父目录存在
    with open(cache_path, "wb") as f:
        pkl.dump(embeddings_tensor, f)

    return embeddings_tensor
