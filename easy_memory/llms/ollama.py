from typing import Dict, List, Optional, Union

from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.configs.llms.ollama import OllamaConfig
from easy_memory.llms.base import LLMBase


class OllamaLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, OllamaConfig, Dict]] = None):
        if config is None:
            config = OllamaConfig()
        elif isinstance(config, dict):
            config = OllamaConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OllamaConfig):
            config = OllamaConfig(
                model=config.model, temperature=config.temperature, api_key=config.api_key,
                max_tokens=config.max_tokens, top_p=config.top_p, top_k=config.top_k,
                enable_vision=config.enable_vision, vision_details=config.vision_details,
                http_client_proxies=getattr(config, 'http_client', None),
            )
        super().__init__(config)

        if not self.config.model:
            self.config.model = "llama3.1"

        import ollama as ollama_lib

        self.client = ollama_lib.Client(
            host=self.config.ollama_base_url or "http://localhost:11434"
        )

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            response_format=None,
            tools: Optional[List[Dict]] = None,
            tool_choice: str = "auto",
            **kwargs,
    ):
        params = {
            "model": self.config.model,
            "messages": messages,
            "options": {"temperature": self.config.temperature},
        }
        if tools:
            params["tools"] = tools

        response = self.client.chat(**params)

        if tools and response.get("message", {}).get("tool_calls"):
            return {
                "content": response["message"]["content"],
                "tool_calls": [
                    {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                    for tc in response["message"]["tool_calls"]
                ],
            }
        return response["message"]["content"]
