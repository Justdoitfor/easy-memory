from typing import Optional
from pydantic import BaseModel, Field

class MilvusDBConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    url: Optional[str] = Field(None, description="Milvus server URL")
    token: Optional[str] = Field(None, description="Token for Milvus server")