from typing import Optional
from pydantic import Field
from easy_memory.configs.rerankers.base import BaseRerankerConfig


class CohereRerankerConfig(BaseRerankerConfig):
    model: Optional[str] = Field(default="rerank-english-v3.0", description="The Cohere rerank model to use")
    return_documents: bool = Field(default=False, description="Whether to return the document texts in the response")
    max_chunks_per_doc: Optional[int] = Field(default=None, description="Maximum number of chunks per document")
