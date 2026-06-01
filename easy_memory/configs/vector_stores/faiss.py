from typing import Optional
from pydantic import BaseModel, Field


class FAISSConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    path: Optional[str] = Field(None, description="Path to store FAISS index and metadata")
