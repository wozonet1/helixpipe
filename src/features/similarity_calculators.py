from Bio import Align
from Bio.Align import substitution_matrices
from joblib import Parallel, delayed
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np
from tqdm import tqdm
from rdkit import Chem


def calculate_drug_similarity(drug_list):
    # 将SMILES转换为RDKit分子对象
    molecules = [Chem.MolFromSmiles(smiles) for smiles in drug_list]
    fpgen = GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [fpgen.GetFingerprint(mol) for mol in molecules]

    num_molecules = len(molecules)
    similarity_matrix = np.zeros((num_molecules, num_molecules))

    # 2. 将外层循环用 tqdm 包裹起来
    # desc 参数为进度条提供了一个清晰的描述
    for i in tqdm(
        range(num_molecules), desc="Calculating Drug Similarities", leave=False
    ):
        # 由于矩阵是对称的，我们可以只计算上三角部分来优化性能
        for j in range(i, num_molecules):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # 计算Tanimoto相似性
                similarity = AllChem.DataStructs.TanimotoSimilarity(
                    fingerprints[i], fingerprints[j]
                )
                # 因为矩阵是对称的，所以同时填充 (i, j) 和 (j, i)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    return similarity_matrix


# region p sim
def align_pair(seq1: str, seq2: str, aligner_config: dict) -> float:
    # 在每个并行进程中重新创建一个 Aligner 对象
    try:
        aligner = Align.PairwiseAligner(**aligner_config)
        return aligner.score(seq1.upper(), seq2.upper())
    except Exception as e:
        print(f"Sequences: {seq1}\n{seq2}")  ##测试为什么用新的align算法会失败
        print(f"Error aligning sequences: {e}")
        return 0.0


def get_custom_blosum62_with_U():
    """
    加载标准的BLOSUM62矩阵,并增加对'U'(硒半胱氨酸)的支持。
    'U'的打分规则将完全模仿'C'(半胱氨酸)。
    """
    # 加载标准的BLOSUM62矩阵
    blosum62 = substitution_matrices.load("BLOSUM62")
    # 将其转换为Python字典以便修改
    custom_matrix_dict = dict(blosum62)
    # 获取所有标准氨基酸的字母 (包括 B, Z, X)
    old_alphabet = blosum62.alphabet
    # 为'U'增加打分规则
    for char in old_alphabet:
        # U-X 的得分 = C-X 的得分

        score = custom_matrix_dict.get(("C", char), custom_matrix_dict.get((char, "C")))
        if score is not None:
            custom_matrix_dict[("U", char)] = score
            custom_matrix_dict[(char, "U")] = score
    # U-U 的得分 = C-C 的得分
    custom_matrix_dict[("U", "U")] = custom_matrix_dict[("C", "C")]
    return substitution_matrices.Array(data=custom_matrix_dict)


def calculate_protein_similarity(sequence_list, cpus: int):
    # 定义 Aligner 的配置字典，以便传递给并行任务s
    aligner_config = {
        "mode": "local",
        "substitution_matrix": get_custom_blosum62_with_U(),
        "open_gap_score": -10,
        "extend_gap_score": -0.5,
    }
    num_sequences = len(sequence_list)
    print("--> Pre-calculating self-alignment scores for normalization...")
    self_scores = Parallel(n_jobs=cpus)(
        delayed(align_pair)(seq, seq, aligner_config)
        for seq in tqdm(
            sequence_list, desc="Self-Alignment Pre-computation", leave=False
        )
    )
    # 加上一个很小的数，防止未来出现除以零的错误 (例如空序列导致score为0)
    self_scores = np.array(self_scores, dtype=np.float32) + 1e-8

    # =========================================================================
    # 3. [两两比对阶段] 生成所有唯一的比对任务并并行计算原始分数
    # =========================================================================
    tasks = []
    for i in range(num_sequences):
        for j in range(i, num_sequences):
            # 我们只计算上三角部分，包括对角线
            tasks.append((i, j))

    print("--> Calculating pairwise raw alignment scores...")
    raw_pairwise_scores = Parallel(n_jobs=cpus)(
        delayed(align_pair)(sequence_list[i], sequence_list[j], aligner_config)
        for i, j in tqdm(tasks, desc="Pairwise Alignment", leave=False)
    )

    # =========================================================================
    # 4. [归一化与填充阶段] 使用预计算的分数进行归一化并构建矩阵
    # =========================================================================
    print("--> Populating similarity matrix with normalized scores...")
    similarity_matrix = np.zeros((num_sequences, num_sequences), dtype=np.float32)

    for (i, j), raw_score in zip(
        tqdm(
            tasks,
            desc="Normalizing and Filling Matrix",
            leave=False,
        ),
        raw_pairwise_scores,
    ):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            # 计算归一化的分母
            denominator = np.sqrt(self_scores[i] * self_scores[j])

            # 应用归一化公式
            normalized_score = raw_score / denominator

            # 使用 np.clip 确保分数严格落在 [0, 1] 区间，处理浮点数精度问题
            final_score = np.clip(normalized_score, 0, 1)

            # 因为矩阵是对称的，所以同时填充 (i, j) 和 (j, i)
            similarity_matrix[i, j] = final_score
            similarity_matrix[j, i] = final_score

    return similarity_matrix


# endregion
