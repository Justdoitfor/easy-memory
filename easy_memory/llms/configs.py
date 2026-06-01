""" LLM的顶层配置验证器。使用Pydantic的 `field_validator` 确保只接受已支持的provider """

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class LlmConfig(BaseModel):
    provider: str = Field(description="Provider of the LLM (e.g. 'ollama','openai') ", default='openai')
    config: Optional[dict] = Field(description="Configuration ofr the speciffic LLM", default={})

    @field_validator('config')
    def validate_config(cls, v, values):
        provider = values.data.get("provider")
        if provider in (
                "openai", "ollama", "anthropic", "groq", "together", "aws_bedrock",
                "litellm", "azure_openai", "openai_structured", "azure_openai_structured",
                "gemini", "deepseek", "minimax", "xai", "sarvam", "lmstudio", "vllm", "langchain",
        ):
            return v
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

