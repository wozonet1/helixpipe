from dataclasses import dataclass, field


@dataclass
class TensorVaultConfig:
    """
    TensorVault 服务的连接与资源映射配置。
    """

    host: str = "localhost:8080"
    enabled: bool = False  # 总开关，方便回退到本地模式

    # 资源映射表 (Logical Name -> Merkle Root Hash)
    dataset_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class StorageConfig:
    """
    顶层存储配置。
    """

    tensorvault: TensorVaultConfig = field(default_factory=TensorVaultConfig)
