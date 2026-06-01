from typing import Optional
from pydantic import BaseModel, Field


class UpstashVectorConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    url: Optional[str] = Field(None, description="URL for Upstash Vector index")
    token: Optional[str] = Field(None, description="Token for Upstash Vector index")
