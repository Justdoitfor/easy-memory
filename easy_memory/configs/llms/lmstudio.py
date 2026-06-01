from typing import Any, Dict, Optional
from easy_memory.configs.llms.base import BaseLlmConfig


class LMStudioConfig(BaseLlmConfig):
    def __init__(self,
                 model=None,
                 temperature=0.1,
                 api_key=None,
                 max_tokens=2000,
                 top_p=0.1,
                 top_k=1,
                 enable_vision=False,
                 vision_details="auto",
                 http_client_proxies=None,
                 lmstudio_base_url=None,
                 lmstudio_response_format=None):
        super().__init__(model=model, temperature=temperature, api_key=api_key, max_tokens=max_tokens,
                         top_p=top_p, top_k=top_k, enable_vision=enable_vision, vision_details=vision_details,
                         http_client_proxies=http_client_proxies)
        self.lmstudio_base_url = lmstudio_base_url or "http://localhost:1234/v1"
        self.lmstudio_response_format = lmstudio_response_format
