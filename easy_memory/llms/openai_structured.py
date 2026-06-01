import os
from typing import Dict, List, Optional

from openai import OpenAI

from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.llms.base import LLMBase


class OpenAIStructuredLLM(LLMBase):
    """使用 OpenAI Structured Outputs 的 LLM 实现。

    与标准 OpenAILLM 的区别：
    - 使用 client.beta.chat.completions.parse() 而非 .create()
    - 原生支持 response_format 和 tools 的结构化输出
    - 更适合需要严格 JSON schema 的场景
    """

    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)
        if not self.config.model:
            self.config.model = "gpt-5-mini"
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.config.openai_base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            response_format: Optional[str] = None,
            tools: Optional[List[Dict]] = None,
            tool_choice: str = "auto",
    ) -> str:
        """生成结构化响应。

        使用 beta.chat.completions.parse() 确保输出严格符合 schema。
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        response = self.client.beta.chat.completions.parse(**params)
        return response.choices[0].message.content
