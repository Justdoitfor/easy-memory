from typing import Optional
from pydantic import BaseModel, Field


class PGVectorConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    user: Optional[str] = Field(None, description="User for PostgreSQL")
    password: Optional[str] = Field(None, description="Password for PostgreSQL")
    host: Optional[str] = Field(None, description="Host for PostgreSQL")
    port: Optional[int] = Field(None, description="Port for PostgreSQL")
    dbname: Optional[str] = Field(None, description="Database name for PostgreSQL")
