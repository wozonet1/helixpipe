from pathlib import Path
from typing import Iterable

import hydra


def ensure_path_exists(filepath: Path) -> None:
    """
    Ensures that the parent directory of a given file path exists.
    If the directory does not exist, it is created.

    Args:
        filepath (Path): The Path object representing the full file path.
    """
    # .parent gets the directory containing the file
    # e.g., for Path('/path/to/my/file.csv'), .parent is Path('/path/to/my')
    parent_directory = filepath.parent

    # .mkdir() creates the directory.
    # `parents=True` means it will create any necessary parent directories as well.
    #   (e.g., if neither '/path' nor '/path/to' exist, it creates both)
    # `exist_ok=True` means it will NOT raise an error if the directory already exists.
    parent_directory.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    【最终版】以一种稳健的方式，获取项目的根目录。
    没有特殊情况,应该尽可能在项目的各个获取最初地址的时候使用这个,依赖项目结构而不是相对地址
    - 当脚本作为Hydra应用运行时，它使用 `hydra.utils.get_original_cwd()` 来
      获取启动命令时所在的原始目录，这能抵抗Hydra工作目录的切换。
    - 当脚本独立运行时 (例如 `python my_script.py`)，`get_original_cwd()`
      会抛出ValueError。在这种情况下，它会回退到使用当前工作目录。

    Returns:
        Path: 指向项目根目录的Path对象。
    """
    try:
        # 路径1: 尝试使用Hydra的官方方法。这是首选，也是最可靠的方法。
        # get_original_cwd() 返回的是您运行 `python src/run.py ...` 命令时
        # 所在的目录，也就是项目根目录。
        project_root = Path(hydra.utils.get_original_cwd())
    except ValueError:
        # 路径2 (备用方案): 如果不在Hydra应用中 (例如，独立运行一个debug脚本)，
        # get_original_cwd() 会失败。此时，我们假定脚本是从项目根目录运行的。
        # Path.cwd() 返回当前工作目录。
        print(
            "Warning: Not running under a Hydra managed process. "
            "Using current working directory as project root. "
            "Ensure you are running the script from the project's root folder."
        )
        project_root = Path.cwd()

    return project_root


def check_paths_exist(paths: Iterable[Path]) -> bool:
    """
    【V2 通用版】检查一个路径迭代器中的所有文件/目录是否存在。

    这个函数是纯粹的、通用的，不依赖任何配置对象或特定的键约定。

    Args:
        paths (Iterable[Path]): 一个包含Path对象的迭代器 (例如, 列表或生成器)。

    Returns:
        bool: 如果所有路径都存在，则返回True；否则返回False。
    """
    for path in paths:
        if not path.exists():
            print(f"--> [Check Failed] Path not found: {path}")
            return False
    return True
