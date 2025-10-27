# 文件: scripts/generate_fake_data.py

import pickle
from pathlib import Path


def create_fake_name_to_cid_pkl():
    """
    为单元测试生成一个伪造的 pubchem_name_to_cid.pkl 文件。
    """
    print("--- Generating fake name-to-CID map for testing ---")

    # 1. 定义数据
    name_to_cid_map = {"aspirin": 201, "caffeine": 202}

    # 2. 定义输出路径
    #    使用 Path(__file__) 可以让脚本在任何位置运行都找到正确的相对路径
    project_root = Path(__file__).resolve().parent.parent
    output_path = (
        project_root
        / "tests"
        / "fake_data_v3"
        / "cache"
        / "ids"
        / "pubchem_name_to_cid.pkl"
    )

    # 3. 确保目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"--> Target directory: {output_path.parent}")

    # 4. 写入 Pickle 文件
    try:
        with open(output_path, "wb") as f:
            pickle.dump(name_to_cid_map, f)
        print(f"✅ Pickle file saved successfully to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving pickle file: {e}")


if __name__ == "__main__":
    create_fake_name_to_cid_pkl()
    # 你可以在这里添加生成其他伪造数据文件的函数
