"""
定义一个包含 错误码、详细信息、修复建议和调试信息 的异常基类
"""

from typing import Any, Dict, Optional


class AgentMemoryError(Exception):
    """Base exception for all memory-related errors"""

    def __init__(self,
                 message: str,
                 error_code: str,
                 details: Optional[Dict[str, Any]] = None,
                 suggestion: Optional[str] = None,
                 debug_info: Optional[Dict[str, Any]] = None,
                 ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.suggestion = suggestion
        self.debug_info = debug_info or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"details={self.details!r}, "
            f"suggestion={self.suggestion!r}, "
            f"debug_info={self.debug_info!r})"
        )


class AuthenticationError(AgentMemoryError):
    pass


class RateLimitError(AgentMemoryError):
    pass


class ValidationError(AgentMemoryError):
    pass


class MemoryNotFoundError(AgentMemoryError):
    pass


class NetworkError(AgentMemoryError):
    pass


class ConfigurationError(AgentMemoryError):
    pass


class MemoryQuotaExceededError(AgentMemoryError):
    pass


class MemoryCorruptionError(AgentMemoryError):
    pass


class VectorSearchError(AgentMemoryError):
    pass


class CacheError(AgentMemoryError):
    pass


class VectorStoreError(AgentMemoryError):
    def __init__(self,
                 message: str,
                 error_code: str = "VECTOR__001",
                 details: dict = None,
                 suggestion: str = "Please check your vector store configuration and connection",
                 debug_info: dict = None,
                 ):
        super().__init__(message, error_code, details, suggestion, debug_info)


class EmbeddingError(AgentMemoryError):
    def __init__(self,
                 message: str,
                 error_code: str = "EMBED_001",
                 details: dict = None,
                 suggestion: str = "Please check your embedding model configuration",
                 debug_info: dict = None):
        super().__init__(message, error_code, details, suggestion, debug_info)


class LLMError(AgentMemoryError):
    def __init__(self,
                 message: str,
                 error_code: str = "LLM_001",
                 details: dict = None,
                 suggestion: str = "Please check your LLM configuration and API key",
                 debug_info: dict = None):
        super().__init__(message, error_code, details, suggestion, debug_info)


class DatabaseError(AgentMemoryError):
    def __init__(self,
                 message: str,
                 error_code: str = "DB_001",
                 details: dict = None,
                 suggestion: str = "Please check your database configuration and connection",
                 debug_info: dict = None):
        super().__init__(message, error_code, details, suggestion, debug_info)


class DependencyError(AgentMemoryError):
    def __init__(self,
                 message: str,
                 error_code: str = "DEPS_001",
                 details: dict = None,
                 suggestion: str = "Please install the required dependencies",
                 debug_info: dict = None):
        super().__init__(message, error_code, details, suggestion, debug_info)


# HTTP状态码对应的异常映射
HTTP_STATUS_TO_EXCEPTION = {
    400: ValidationError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: MemoryNotFoundError,
    408: NetworkError,
    409: ValidationError,
    413: MemoryQuotaExceededError,
    422: ValidationError,
    429: RateLimitError,
    500: AgentMemoryError,
    502: NetworkError,
    503: NetworkError,
    504: NetworkError,
}


def create_exception_from_response(
        status_code: int,
        response_text: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        debug_info: Optional[Dict[str, Any]] = None,
) -> AgentMemoryError:
    exception_class = HTTP_STATUS_TO_EXCEPTION.get(status_code, AgentMemoryError)
    if not error_code:
        error_code = f"HTTP_{status_code}"
    suggestions = {
        400: "Please check your request parameters and try again",
        401: "Please check your API key and authentication credentials",
        403: "You don't have permission to perform this operation",
        404: "The requested resource was not found",
        408: "Request timed out. Please try again",
        409: "Resource conflict. Please check your request",
        413: "Request too large. Please reduce the size of your request",
        422: "Invalid request data. Please check your input",
        429: "Rate limit exceeded. Please wait before making more requests",
        500: "Internal server error. Please try again later",
        502: "Service temporarily unavailable. Please try again later",
        503: "Service unavailable. Please try again later",
        504: "Gateway timeout. Please try again later",
    }
    return exception_class(
        message=response_text or f"HTTP {status_code} error",
        error_code=error_code,
        details=details or {},
        suggestion=suggestions.get(status_code, "Please try again later"),
        debug_info=debug_info or {}
    )
