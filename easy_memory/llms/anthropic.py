import os
from typing import Dict, List, Optional, Union

from easy_memory.configs.llms.anthropic import AnthropicConfig
from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.llms.base import LLMBase


class AnthropicLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, AnthropicConfig, Dict]] = None):
        if config is None:
            config = AnthropicConfig()
        elif isinstance(config, dict):
            config = AnthropicConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, AnthropicConfig):
            config = AnthropicConfig(
                model=config.model, temperature=config.temperature, api_key=config.api_key,
                max_tokens=config.max_tokens, top_p=config.top_p, top_k=config.top_k,
                enable_vision=config.enable_vision, vision_details=config.vision_details,
                http_client_proxies=getattr(config, 'http_client', None),
            )
        super().__init__(config)

        if not self.config.model:
            self.config.model = "claude-3-5-sonnet-latest"

        import anthropic

        kwargs = {"api_key": self.config.api_key or os.getenv("ANTHROPIC_API_KEY")}
        if self.config.anthropic_base_url:
            kwargs["base_url"] = self.config.anthropic_base_url
        self.client = anthropic.Anthropic(**kwargs)

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
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        if tools:
            params["tools"] = tools

        response = self.client.messages.create(**params)

        if tools and response.content:
            processed = {"content": "", "tool_calls": []}
            for block in response.content:
                if hasattr(block, "text"):
                    processed["content"] += block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    processed["tool_calls"].append(
                        {"name": block.name, "arguments": block.input}
                    )
            return processed
        return response.content[0].text if response.content else ""
