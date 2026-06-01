from typing import Optional
from pydantic import BaseModel, Field


class PineconeConfig(BaseModel):
    collection_name: Optional[str] = Field("easy-memory", description="Name of the collection (index)")
    api_key: Optional[str] = Field(None, description="API key for Pinecone")
    environment: Optional[str] = Field(None, description="Environment for Pinecone")
    server_url: Optional[str] = Field(None, description="Server URL for Pinecone")
    metric: Optional[str] = Field("cosine", description="Distance metric")
    batch_size: Optional[int] = Field(100, description="Batch size for operations")
