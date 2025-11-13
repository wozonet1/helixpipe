import json
from pathlib import Path


def create_fake_brenda_json():
    """
    为 BrendaProcessor 的单元测试，生成一个最小化的、覆盖所有关键场景的
    伪造 BRENDA JSON 文件。
    """
    print("--- Generating fake BRENDA JSON file for testing ---")

    # --- 设计伪造数据 ---
    # 我们将构建一个包含多个EC条目的字典
    fake_data = {
        # --- 场景1: 一个理想的、包含所有信息的EC条目 ---
        "1.1.1.1": {
            "protein": {
                "1": {  # 内部protein ID
                    "organism": "Homo sapiens",
                    "accessions": ["P01_A", "P01_B"],  # 包含UniProt ID
                    "source": "SWISS-PROT",
                },
                "2": {
                    "organism": "Mus musculus",  # 非人类，应该被忽略
                    "accessions": ["MOUSE_P"],
                },
                "3": {
                    "organism": "Homo sapiens",
                    "accessions": [],  # accessions为空，应该被忽略
                },
            },
            # a. 匹配的 inhibitor
            "inhibitor": [
                {
                    "value": "aspirin",  # 将被映射为CID 201
                    "proteins": ["1"],
                    "comment": "# ... #",
                }
            ],
            # b. 匹配的 Ki 值
            "ki_value": [
                {
                    "value": "80.0 {caffeine}",  # 值低于阈值，应该保留。将被映射为CID 202
                    "proteins": ["1"],
                    "comment": "# pH 7.4 #",
                }
            ],
            # c. Ki 值过高，应该被过滤
            "ki_value:1": [  # BRENDA中常见的重复键
                {
                    "value": "15000.0 {theophylline}",  # 值高于默认阈值(10000)
                    "proteins": ["1"],
                    "comment": "# ... #",
                }
            ],
        },
        # --- 场景2: 一个只包含 Km 值的EC条目 ---
        "2.7.1.1": {
            "protein": {
                "4": {
                    "organism": "Homo sapiens",
                    "accessions": ["P02"],
                }
            },
            # a. 匹配的 Km 值
            "km_value": [
                {
                    "value": "0.1 {glucose}",  # 假设单位是mM, 换算后为100000nM，会被过滤
                    "proteins": ["4"],
                }
            ],
            # b. 另一个 Km 值，会被保留
            "km_value:1": [
                {
                    "value": "0.005 {ATP}",  # 换算后为5000nM, 应该保留
                    "proteins": ["4"],
                }
            ],
        },
        # --- 场景3: 一个没有人类蛋白质的EC条目，应该被完全忽略 ---
        "3.1.1.1": {
            "protein": {
                "5": {"organism": "Escherichia coli", "accessions": ["ECOLI_P"]}
            },
            "inhibitor": [{"value": "some_inhibitor", "proteins": ["5"]}],
        },
    }

    # 包装成BRENDA的顶层结构
    final_json_structure = {"data": fake_data}

    # --- 定义输出路径 ---
    # 使用 Path(__file__) 可以让脚本在任何位置运行都找到正确的相对路径
    project_root = Path(__file__).resolve().parents[2]  # 假设脚本在 tests/scripts/
    output_path = (
        project_root
        / "tests"
        / "fake_data_v3"
        / "brenda"
        / "raw"
        / "brenda_2025_1.json"
    )

    # --- 确保目录存在并写入文件 ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"--> Target directory: {output_path.parent}")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_json_structure, f, indent=4)
        print(f"✅ Fake BRENDA JSON file saved successfully to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")

    # --- [额外] 生成一个配套的名称到CID的映射文件 ---
    # 这对于测试 BrendaProcessor 的名称映射逻辑至关重要
    name_to_cid_map = {
        "aspirin": 201,
        "caffeine": 202,
        "theophylline": 203,
        "glucose": 204,
        "atp": 205,
    }

    map_output_path = (
        project_root
        / "tests"
        / "fake_data_v3"
        / "cache"
        / "ids"
        / "pubchem_name_to_cid.pkl"
    )
    map_output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pickle

        with open(map_output_path, "wb") as f:
            pickle.dump(name_to_cid_map, f)
        print(f"✅配套的 fake name-to-CID map 已保存到: {map_output_path}")
    except Exception as e:
        print(f"❌ Error saving pickle file for name map: {e}")


if __name__ == "__main__":
    create_fake_brenda_json()
