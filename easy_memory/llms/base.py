""" 所有LLM提供商的基类 """

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from easy_memory.configs.llms.base import BaseLlmConfig


class LLMBase(ABC):
    """Base class for all LLM Providers."""

    def __init__(self, config: Optional[Union[BaseLlmConfig, Dict]] = None):
        if config is None:
            self.config = BaseLlmConfig()
        elif isinstance(config, dict):
            self.config = BaseLlmConfig(**config)
        else:
            self.config = config
        self._validate_config()

    def _validate_config(self):
        if not hasattr(self.config, 'model'):
            raise ValueError("Config must have 'model' attribute")

    def _is_reasoning_model(self, model: str) -> bool:
        reasoning_models = {
            "o1", "o1-preview", "o3-mini", "o3",
            "gpt-5", "gpt-5o", "gpt-5o-mini", "gpt-5o-micro",
        }
        base_model = model.lower().rsplit("/", 1)[-1]
        if base_model in reasoning_models:
            return True
        if any(base_model.startswith(prefix) for prefix in ["o1-", "o1.", "o3-", "o3."]):
            return True
        return False

    def _get_supported_params(self, **kwargs) -> Dict:
        model = getattr(self.config, 'model', '')
        if self._is_reasoning_model(model):
            param_keys = ["messages", "response_format", "tools", "tool_choice"]
            supported_params = {k: kwargs.get(k) for k in param_keys if k in kwargs}
            reasoning_effort = getattr(self.config, 'reasoning_effort', None)
            if reasoning_effort:
                supported_params["reasoning_effort"] = reasoning_effort
            return supported_params
        else:
            return self._get_common_params(**kwargs)

    def _get_common_params(self, **kwargs) -> Dict:
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        params.update(kwargs)
        return params

    @abstractmethod
    def generate_response(self,
                          messages: List[Dict[str,str]],
                          tools: Optional[List[Dict]] = None,
                          tool_choice: Optional[str] = "auto",
                          **kwargs
                          ):
        pass
