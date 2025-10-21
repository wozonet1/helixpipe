# 文件: src/configs/analysis.py (全新)

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class PlotSettingsConfig:
    """
    【结构化配置】定义所有绘图共享的默认设置。
    """

    style: str = "seaborn-v0_8-whitegrid"
    # figure_size 可以是整数列表，也可以是浮点数列表
    figure_size: List[Union[int, float]] = field(default_factory=lambda: [12, 8])
    dpi: int = 300
    palette: str = "muted"
    # 允许有额外的、未在上面定义的字段（例如'bins'）
    # 这是通过在主dataclass中使用一个开放的字典来实现的，
    # 或者我们可以直接在这里添加 'bins' 字段并设为可选。
    bins: Optional[int] = None


@dataclass
class AnalysisConfig:
    """
    【结构化配置】定义一个分析任务的配置。
    """

    # 任务名称，没有默认值，强制每个分析任务都必须定义它
    task_name: str = "default_task_name"

    # 嵌套的 plot_settings
    plot_settings: PlotSettingsConfig = field(default_factory=PlotSettingsConfig)
