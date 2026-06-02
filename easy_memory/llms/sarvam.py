import os
from typing import Dict, List, Optional

from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.llms.base import LLMBase


class SarvamLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "sarvam-2b-v0.5"

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.config.api_key or os.getenv("SARVAM_API_KEY"),
            base_url="https://api.sarvam.ai/v1",
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

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
