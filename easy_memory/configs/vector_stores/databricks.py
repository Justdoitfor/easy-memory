from typing import Optional
from pydantic import BaseModel, Field


class DatabricksConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    endpoint: Optional[str] = Field(None, description="Databricks workspace URL")
    token: Optional[str] = Field(None, description="Personal access token for authentication")
    catalog: Optional[str] = Field(None, description="Unity Catalog catalog name")
    schema: Optional[str] = Field(None, description="Unity Catalog schema name")
