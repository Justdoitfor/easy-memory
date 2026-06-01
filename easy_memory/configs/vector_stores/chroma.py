from typing import Optional
from pydantic import BaseModel, Field


class ChromaDbConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    hosting_url: Optional[str] = Field(None, description="Hosting URL for the Chroma server")
    path: Optional[str] = Field("/tmp/chroma_db", description="Path for local Chroma database")
