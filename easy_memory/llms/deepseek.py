import os
from typing import Dict, List, Optional, Union

from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.configs.llms.deepseek import DeepSeekConfig
from easy_memory.llms.base import LLMBase


class DeepSeekLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, DeepSeekConfig, Dict]] = None):
        if config is None:
            config = DeepSeekConfig()
        elif isinstance(config, dict):
            config = DeepSeekConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, DeepSeekConfig):
            config = DeepSeekConfig(
                model=config.model, temperature=config.temperature, api_key=config.api_key,
                max_tokens=config.max_tokens, top_p=config.top_p, top_k=config.top_k,
            )
        super().__init__(config)

        if not self.config.model:
            self.config.model = "deepseek-chat"

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.config.api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.config.deepseek_base_url or "https://api.deepseek.com/v1",
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
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if response_format:
            params["response_format"] = response_format

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
