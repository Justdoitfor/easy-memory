from typing import Optional
from pydantic import BaseModel, Field


class OpenSearchConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    host: Optional[str] = Field(None, description="OpenSearch host")
    port: Optional[int] = Field(None, description="OpenSearch port")
    api_key: Optional[str] = Field(None, description="OpenSearch API key")
