from typing import Any, Dict, Optional

from easy_memory.configs.base import AzureConfig
from easy_memory.configs.llms.base import BaseLlmConfig


class AzureOpenAIConfig(BaseLlmConfig):
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        reasoning_effort: Optional[str] = None,
        http_client_proxies: Optional[dict] = None,
        azure_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model=model, temperature=temperature, api_key=api_key, max_tokens=max_tokens,
            top_p=top_p, top_k=top_k, enable_vision=enable_vision, vision_details=vision_details,
            reasoning_effort=reasoning_effort, http_client_proxies=http_client_proxies,
        )
        self.azure_kwargs = AzureConfig(**(azure_kwargs or {}))