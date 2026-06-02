import os
from typing import Dict, List, Optional, Union

from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.configs.llms.lmstudio import LMStudioConfig
from easy_memory.llms.base import LLMBase


class LMStudioLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, LMStudioConfig, Dict]] = None):
        if config is None:
            config = LMStudioConfig()
        elif isinstance(config, dict):
            config = LMStudioConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, LMStudioConfig):
            config = LMStudioConfig(
                model=config.model, temperature=config.temperature, api_key=config.api_key,
                max_tokens=config.max_tokens, top_p=config.top_p, top_k=config.top_k,
            )
        super().__init__(config)

        if not self.config.model:
            self.config.model = "llama-3.2-1b-instruct"

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.config.api_key or "lm-studio",
            base_url=self.config.lmstudio_base_url,
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
