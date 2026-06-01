from abc import ABC
from typing import Dict, Optional, Union
import httpx


class BaseLlmConfig(ABC):
    def __init__(self,
                 model: Optional[Union[str, Dict]] = None,
                 temperature: float = 0.1,
                 api_key: Optional[str] = None,
                 max_tokens: int = 2000,
                 top_p: float = 0.7,
                 top_k: int = 1,
                 enable_vision: bool = False,
                 vision_details: Optional[str] = 'auto',
                 reasoning_effort: Optional[str] = None,
                 http_client_proxies: Optional[Union[Dict, str]] = None,
                 ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.enable_vision = enable_vision
        self.vision_details = vision_details
        self.reasoning_effort = reasoning_effort
        self.http_client_proxies = httpx.Client(proxy=http_client_proxies) if http_client_proxies else None

