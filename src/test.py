from Bio.Align import substitution_matrices


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
    new_alphabet = old_alphabet + "U"
    # 为'U'增加打分规则
    for char in old_alphabet:
        # U-X 的得分 = C-X 的得分

        score = custom_matrix_dict.get(("C", char), custom_matrix_dict.get((char, "C")))
        if score is not None:
            custom_matrix_dict[("U", char)] = score
            custom_matrix_dict[(char, "U")] = score
    # U-U 的得分 = C-C 的得分
    custom_matrix_dict[("U", "U")] = custom_matrix_dict[("C", "C")]
    final_matrix = substitution_matrices.Array(data=custom_matrix_dict)
    return final_matrix


print(get_custom_blosum62_with_U())
