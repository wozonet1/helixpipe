import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import pickle as pkl
import warnings
from transformers import EsmModel

# 导入您自己的工具包，用于确保路径存在
from research_template import ensure_path_exists


# ------------------- 模型加载与初始化 -------------------


def get_esm_model_and_alphabet(
    model_name: str = "facebook/esm2_t30_150M_UR50D", device: str = "cpu"
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

    # a. 加载模型和分词器
    tokenizer, model = get_esm_model_and_alphabet(model_name, device)

    all_embeddings = []
    progress_bar = tqdm(
        range(0, len(sequences), batch_size),
        desc="[ESM Extractor] Online Protein Batches",
    )

    for i in progress_bar:
        batch_seqs = sequences[i : i + batch_size]

        # [修改] 使用Hugging Face tokenizer进行编码
        inputs = tokenizer(
            batch_seqs, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            # [修改] Hugging Face模型的输出是一个字典
            outputs = model(**inputs)
            # 提取最后一层的隐藏状态
            token_representations = outputs.last_hidden_state

        # [修改] 后处理逻辑，需要考虑padding
        for j, seq_string in enumerate(batch_seqs):
            # 获取attention_mask来确定哪些是真实的token，哪些是padding
            attention_mask = inputs["attention_mask"][j]
            true_token_reprs = token_representations[j][attention_mask == 1]
            # 去掉开头的<cls>和结尾的<eos>
            true_token_reprs = true_token_reprs[1:-1]

            sequence_level_repr = true_token_reprs.mean(dim=0)
            all_embeddings.append(sequence_level_repr.cpu())

    embeddings_tensor = torch.stack(all_embeddings)
    # c. 保存到缓存以备后用
    print(
        f"--> [ESM Extractor] Saving newly generated embeddings to cache: {cache_path}"
    )
    ensure_path_exists(cache_path)  # 确保父目录存在
    with open(cache_path, "wb") as f:
        pkl.dump(embeddings_tensor, f)

    return embeddings_tensor


def get_chemberta_model_and_tokenizer(
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


def extract_chemberta_molecule_embeddings(
    smiles_list: list[str],
    cache_path: Path,
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 64,  # 分子通常更短，可以用更大的batch_size
    device: str = "cpu",
    force_regenerate: bool = False,
) -> torch.Tensor:
    """
    【核心接口】智能地提取或加载小分子（SMILES）嵌入。
    """
    if cache_path.exists() and not force_regenerate:
        print(
            f"--> [ChemBERTa Extractor] Loading cached molecule embeddings from: {cache_path}"
        )
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    if force_regenerate:
        print(
            "--> [ChemBERTa Extractor] `force_regenerate` is True. Starting online extraction..."
        )
    else:
        print(
            f"--> [ChemBERTa Extractor] No cache found at '{cache_path}'. Starting online extraction..."
        )

    if not smiles_list:
        warnings.warn("Input SMILES list is empty. Returning an empty tensor.")
        temp_model, _ = get_chemberta_model_and_tokenizer(model_name, device)
        embedding_dim = temp_model.config.hidden_size
        del temp_model
        return torch.empty((0, embedding_dim))

    tokenizer, model = get_chemberta_model_and_tokenizer(model_name, device)

    all_embeddings = []
    progress_bar = tqdm(
        range(0, len(smiles_list), batch_size),
        desc="[ChemBERTa Extractor] Online Molecule Batches",
    )

    for i in progress_bar:
        batch_smiles = smiles_list[i : i + batch_size]

        inputs = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )  # 设定一个最大长度
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # 我们使用 [CLS] token 的输出作为整个分子的表示，这是BERT类模型的标准做法
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(batch_embeddings.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)

    print(
        f"--> [ChemBERTa Extractor] Saving newly generated embeddings to cache: {cache_path}"
    )
    ensure_path_exists(cache_path)
    with open(cache_path, "wb") as f:
        pkl.dump(embeddings_tensor, f)

    return embeddings_tensor
