from typing import Optional, List, Callable, Any
from easy_memory.configs.llms.base import BaseLlmConfig


class OpenAIConfig(BaseLlmConfig):
    def __init__(
            self,
            model: Optional[str] = None,
            temperature: float = 0.1,
            api_key: Optional[str] = None,
            max_tokens: int = 2000,
            top_p: float = 0.1,
            top_k: int = 1,
            enable_vision: bool = False,
            vision_details: Optional[str] = 'auto',
            reasoning_effort: Optional[str] = None,
            http_client_proxies: Optional[dict] = None,
            openai_base_url: Optional[str] = None,
            models: Optional[List[str]] = None,
            route: Optional[str] = 'fallback',
            openrouter_base_url: Optional[str] = None,
            site_url: Optional[str] = None,
            app_name: Optional[str] = None,
            store: Optional[str] = None,
            response_callback: Optional[Callable[[Any, dict, dict], None]] = None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            reasoning_effort=reasoning_effort,
            http_client_proxies=http_client_proxies,
        )
        self.openai_base_url = openai_base_url
        self.models = models
        self.route = route
        self.openrouter_base_url = openrouter_base_url
        self.site_url = site_url
        self.app_name = app_name
        self.store = store
        self.response_callback = response_callback
