from typing import Optional
from pydantic import BaseModel, Field


class GoogleMatchingEngineConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    project_id: Optional[str] = Field(None, description="Google Cloud project ID")
    location: Optional[str] = Field(None, description="Google Cloud region")
    index_id: Optional[str] = Field(None, description="Vertex AI Vector Search index ID")
    endpoint_id: Optional[str] = Field(None, description="Vertex AI Vector Search endpoint ID")
