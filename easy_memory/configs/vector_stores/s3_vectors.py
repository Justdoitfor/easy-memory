from typing import Optional
from pydantic import BaseModel, Field


class S3VectorsConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    bucket: Optional[str] = Field(None, description="S3 bucket name")
    region: Optional[str] = Field(None, description="AWS region")
