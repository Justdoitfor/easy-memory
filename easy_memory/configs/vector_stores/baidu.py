from typing import Optional
from pydantic import BaseModel, Field


class BaiduDBConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    host: Optional[str] = Field(None, description="Host address for Baidu VectorDB")
    port: Optional[int] = Field(None, description="Port for Baidu VectorDB")
    user: Optional[str] = Field(None, description="User for Baidu VectorDB")
    password: Optional[str] = Field(None, description="Password for Baidu VectorDB")
