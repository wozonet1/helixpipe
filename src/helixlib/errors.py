# research_template/errors.py

from typing import Union

from omegaconf.errors import OmegaConfBaseException


class ConfigPathError(KeyError):
    """
    一个自定义异常，用于在配置中找不到路径或相关键时提供清晰的、结构化的错误信息。
    """

    def __init__(
        self,
        message: str,
        file_key: str,
        failed_interpolation_key: Union[str, None] = None,
        original_exception: Union[Exception, None] = None,
    ):
        """
        初始化异常对象，存储所有相关的上下文信息。
        """
        self.message = message
        self.file_key = file_key
        self.failed_interpolation_key = failed_interpolation_key
        self.original_exception = original_exception

        # 调用super().__init__仍然是好习惯，但我们不再依赖它来格式化
        super().__init__(message)

    def __str__(self) -> str:
        """
        【核心】重写 __str__ 方法，以完全控制错误信息的最终格式。
        """
        header = "\n\n========================= CONFIGURATION PATH ERROR ========================="
        footer = "==========================================================================\n"

        # 1. 构建核心错误信息
        error_line = f"❌ Error: {self.message}"

        # 2. 【智能分析】构建具体原因
        if isinstance(self.original_exception, OmegaConfBaseException):
            # 如果是 OmegaConf 的特定异常，我们进行结构化展示
            reason_line = (
                f"❓ Reason: An OmegaConf error occurred during path resolution for key '{self.file_key}':\n"
                f"   - Error Type: {type(self.original_exception).__name__}\n"
                f"   - Details: {str(self.original_exception).splitlines()[0]}"
            )
        elif self.failed_interpolation_key:
            # 如果提供了失败的插值键
            reason_line = f"❓ Reason: Failed to resolve interpolation for key '{self.failed_interpolation_key}' used in path '{self.file_key}'."
        else:
            # 默认的简单情况
            reason_line = (
                f"❓ Reason: The required key '{self.file_key}' could not be resolved."
            )

        # 3. 构建通用的排错提示
        tips = (
            "💡 Troubleshooting Tips:\n"
            "   1. Check for typos in your '.yaml' files or command-line overrides.\n"
            "   2. Ensure all required config keys are defined and accessible in the current context.\n"
            "   3. Verify that all config interpolations (e.g., `${...}`) are valid."
        )

        # 4. 组合所有部分
        return f"{header}\n{error_line}\n{reason_line}\n\n{tips}\n{footer}"


class SchemaRegistrationError(Exception):
    """
    当在ConfigStore中注册结构化配置Schema失败时抛出的自定义异常。
    """

    def __init__(self, schema_name: str, original_exception: Exception) -> None:
        self.schema_name = schema_name
        self.original_exception = original_exception

        # 构造一个更清晰、更友好的错误消息
        message = (
            f"\n❌ Schema Registration Failed for '{self.schema_name}'!\n"
            f"   This is likely due to a type mismatch or an invalid default value in your dataclass definition.\n"
            f"\n   --- Original Error ---\n"
            f"   Type: {type(self.original_exception).__name__}\n"
            f"   Message: {self.original_exception}\n"
            f"   ------------------------\n"
            f"   Please check the definition of the '{self.schema_name}' dataclass and its nested components."
        )

        # 调用父类的构造函数来设置错误消息
        super().__init__(message)
