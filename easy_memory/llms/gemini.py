import os
from typing import Dict, List, Optional

from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.llms.base import LLMBase


class GeminiLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gemini-2.0-flash-001"

        from google import genai

        self.client = genai.Client(
            api_key=self.config.api_key or os.getenv("GOOGLE_API_KEY")
        )

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            response_format=None,
            tools: Optional[List[Dict]] = None,
            tool_choice: str = "auto",
            **kwargs,
    ):
        from google.genai import types

        contents = []
        for msg in messages:
            if msg["role"] == "user":
                contents.append(
                    types.Content(
                        role="user", parts=[types.Part(text=msg["content"])]
                    )
                )
            elif msg["role"] == "assistant":
                contents.append(
                    types.Content(
                        role="model", parts=[types.Part(text=msg["content"])]
                    )
                )

        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )
        response = self.client.models.generate_content(
            model=self.config.model, contents=contents, config=config
        )
        return response.text or ""
