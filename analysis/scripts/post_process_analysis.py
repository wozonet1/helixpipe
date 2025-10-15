import pandas as pd
from pathlib import Path
import sys

# 将 plot_utils 添加到路径
sys.path.append(str(Path(__file__).parent))
from plot_utils import plot_grouped_bar_chart

def collate_and_plot(multirun_dir: Path, dataset_name: str):
    """
    收集Hydra多任务运行的所有分析结果，并生成最终的对比图。
    """
    print(f"\n--- Post-processing analysis results in: {multirun_dir} ---")
    
    # 1. 搜索并合并所有 'mean_edge_counts.csv' 文件
    all_results = []
    for csv_path in sorted(multirun_dir.glob("**/mean_edge_counts.csv")):
        all_results.append(pd.read_csv(csv_path))
        
    if not all_results:
        print("❌ ERROR: No 'mean_edge_counts.csv' files found in the multirun directory.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    
    # 2. 调用绘图函数
    output_path = multirun_dir.parent / f"{dataset_name}_edge_structure_comparison.png"
    
    plot_grouped_bar_chart(
        df=final_df,
        index_col="config_name",
        columns_col="edge_type",
        values_col="count",
        title=f"Graph Edge Structure Comparison for '{dataset_name.upper()}'",
        xlabel="Configuration (Data Params + Relations)",
        ylabel="Average Number of Edges (Log Scale)",
        output_path=output_path
    )
    print(f"\n✅ Final comparison plot saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python post_process_analysis.py <path_to_hydra_multirun_dir> <dataset_name>")
        sys.exit(1)
        
    multirun_path = Path(sys.argv[1])
    dataset = sys.argv[2]
    collate_and_plot(multirun_path, dataset)