# 文件：src/features/__init__.py

# 导入整个子模块，并给予一个清晰、简洁的别名
from . import feature_extractors as extractors
from . import similarity_calculators as sim_calculators

# [可选] 如果有一些极度常用的函数，希望可以直接通过 features.xxx 访问，
# 可以在这里“暴露”出来。
# 例如，如果我们总是需要用到esm提取器：
# from .feature_extractors import extract_esm_protein_embeddings

# 但通常，保持两层命名空间（features.extractors）是更清晰的做法。
