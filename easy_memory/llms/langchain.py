from typing import Dict, List, Optional

from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.llms.base import LLMBase


class LangchainLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            response_format=None,
            tools: Optional[List[Dict]] = None,
            tool_choice: str = "auto",
            **kwargs,
    ):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        llm = self.config.model
        response = llm.invoke(lc_messages)
        return response.content
