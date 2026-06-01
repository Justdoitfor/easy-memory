from typing import Optional
from pydantic import BaseModel, Field


class AzureMySQLConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    host: Optional[str] = Field(None, description="MySQL server host")
    port: Optional[int] = Field(None, description="MySQL server port")
    user: Optional[str] = Field(None, description="Database user")
    password: Optional[str] = Field(None, description="Database password")
    database: Optional[str] = Field(None, description="Database name")
