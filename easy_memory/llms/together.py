import os
from typing import Dict, List, Optional

from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.llms.base import LLMBase


class TogetherLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "llama3.1-70b"

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.config.api_key or os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
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
