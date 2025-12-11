import logging
from typing import Optional

from tensorvault import Client

from helixpipe.typing import AppConfig

logger = logging.getLogger(__name__)


class TVManager:
    """
    TensorVault 客户端的全局单例管理器。
    确保整个生命周期内只建立一个 gRPC Channel。
    """

    _instance: Optional[Client] = None

    @classmethod
    def get_client(cls, config: AppConfig) -> Client:
        if cls._instance is None:
            host = config.storage.tensorvault.host
            logger.info(f"🔌 Connecting to TensorVault at {host}...")
            cls._instance = Client(addr=host)
        return cls._instance

    @classmethod
    def close(cls):
        if cls._instance:
            logger.info("🔌 Closing TensorVault connection...")
            cls._instance.close()
            cls._instance = None
