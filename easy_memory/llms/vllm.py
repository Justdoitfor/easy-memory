import os
from typing import Dict, List, Optional, Union

from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.configs.llms.vllm import VllmConfig
from easy_memory.llms.base import LLMBase


class VllmLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, VllmConfig, Dict]] = None):
        if config is None:
            config = VllmConfig()
        elif isinstance(config, dict):
            config = VllmConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, VllmConfig):
            config = VllmConfig(
                model=config.model, temperature=config.temperature, api_key=config.api_key,
                max_tokens=config.max_tokens, top_p=config.top_p, top_k=config.top_k,
            )
        super().__init__(config)

        if not self.config.model:
            self.config.model = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ"

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.config.api_key or "EMPTY",
            base_url=self.config.vllm_base_url,
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
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
