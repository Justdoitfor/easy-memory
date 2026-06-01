from typing import Optional
from pydantic import BaseModel, Field


class NeptuneAnalyticsConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    host: Optional[str] = Field(None, description="Neptune host")
    port: Optional[int] = Field(None, description="Neptune port")
