from typing import Any, Dict, Optional
from pydantic import Field
from easy_memory.configs.rerankers.base import BaseRerankerConfig


class LLMRerankerConfig(BaseRerankerConfig):
    model: str = Field(default="gpt-4o-mini", description="LLM model to use for reranking")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, etc.)")
    top_k: Optional[int] = Field(default=None, description="Number of top documents to return after reranking")
    temperature: float = Field(default=0.0, description="Temperature for LLM generation")
    max_tokens: int = Field(default=100, description="Maximum tokens for LLM response")
    scoring_prompt: Optional[str] = Field(default=None, description="Custom prompt template for scoring documents")
    llm: Optional[Dict[str, Any]] = Field(default=None, description="Nested LLM configuration")
