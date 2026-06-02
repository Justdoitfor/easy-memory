import os
from typing import Dict, List, Optional, Union

from easy_memory.configs.llms.azure import AzureOpenAIConfig
from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.llms.base import LLMBase


class AzureOpenAILLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, AzureOpenAIConfig, Dict]] = None):
        if config is None:
            config = AzureOpenAIConfig()
        elif isinstance(config, dict):
            config = AzureOpenAIConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, AzureOpenAIConfig):
            config = AzureOpenAIConfig(
                model=config.model, temperature=config.temperature, api_key=config.api_key,
                max_tokens=config.max_tokens, top_p=config.top_p, top_k=config.top_k,
            )
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4o-mini"

        from openai import AzureOpenAI

        azure = self.config.azure_kwargs
        self.client = AzureOpenAI(
            api_key=azure.api_key or self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=azure.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=azure.api_version or os.getenv("OPENAI_API_VERSION", "2024-02-01"),
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
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
