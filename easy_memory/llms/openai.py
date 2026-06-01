import json
import logging
import os
from typing import Dict, List, Optional, Union

from openai import OpenAI

from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.configs.llms.openai import OpenAIConfig
from easy_memory.llms.base import LLMBase


def extract_json(text):
    """从文本中提取 JSON，处理代码块和裸 JSON。

    LLM 返回的内容可能包含：
    - 纯 JSON: {"key": "value"}
    - 代码块包裹: ```json\n{"key": "value"}\n```
    - 思考标签: <think>...</think> 前缀

    此函数清理这些格式并提取有效的 JSON。
    """
    import re
    # 移除代码块标记
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    # 移除思考标签（某些模型如 DeepSeek 会输出）
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()
    # 尝试直接解析
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        # 尝试提取花括号包裹的 JSON 对象
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group()
        # 尝试提取方括号包裹的 JSON 数组
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return match.group()
    return text


class OpenAILLM(LLMBase):
    """OpenAI 兼容的 LLM 实现。

    支持：
    - OpenAI 官方 API
    - OpenRouter（通过 OPENROUTER_API_KEY 环境变量）
    - 任何 OpenAI 兼容的 API（通过自定义 base_url）
    - Tool calling（函数调用）
    - Response format（结构化输出）
    """

    def __init__(self, config: Optional[Union[BaseLlmConfig, OpenAIConfig, Dict]] = None):
        # 配置标准化：确保 config 是 OpenAIConfig 类型
        if config is None:
            config = OpenAIConfig()
        elif isinstance(config, dict):
            config = OpenAIConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OpenAIConfig):
            # 从基础配置升级为 OpenAI 专有配置
            config = OpenAIConfig(
                model=config.model, temperature=config.temperature, api_key=config.api_key,
                max_tokens=config.max_tokens, top_p=config.top_p, top_k=config.top_k,
                enable_vision=config.enable_vision, vision_details=config.vision_details,
                reasoning_effort=getattr(config, 'reasoning_effort', None),
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        # 默认模型
        if not self.config.model:
            self.config.model = "gpt-5-mini"

        # 根据环境变量选择 API 端点
        if os.environ.get("OPENROUTER_API_KEY"):
            # OpenRouter 模式：使用 OpenRouter 的 API
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                         or os.getenv("OPENROUTER_API_BASE")
                         or "https://openrouter.ai/api/v1",
            )
        else:
            # 标准 OpenAI 模式
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = self.config.openai_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _parse_response(self, response, tools):
        """解析 OpenAI API 响应。

        如果使用了 tools（函数调用），提取工具调用信息；
        否则直接返回文本内容。
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )
            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            response_format=None,
            tools: Optional[List[Dict]] = None,
            tool_choice: str = "auto",
            **kwargs,
    ):
        """生成 LLM 响应。

        这是 LLMBase 定义的核心接口方法。

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            response_format: 响应格式约束（如 JSON schema）
            tools: 工具定义列表（用于 function calling）
            tool_choice: 工具选择策略 ("auto", "none", "required")
            **kwargs: 额外参数

        Returns:
            普通模式：str（响应文本）
            工具模式：dict（包含 content 和 tool_calls）
        """
        # 获取模型支持的参数
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update({"model": self.config.model, "messages": messages})

        # OpenRouter 特殊参数
        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            if self.config.models:
                openrouter_params["models"] = self.config.models
                openrouter_params["route"] = self.config.route
                params.pop("model")
            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers
            params.update(**openrouter_params)
        else:
            if self.config.store is not None:
                params["store"] = self.config.store

        # 添加可选参数
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        # 调用 API
        response = self.client.chat.completions.create(**params)
        parsed_response = self._parse_response(response, tools)

        # 执行响应回调（用于日志、监控等）
        if self.config.response_callback:
            try:
                self.config.response_callback(self, response, params)
            except Exception as e:
                logging.error(f"Error due to callback: {e}")

        return parsed_response
