from typing import Optional
from pydantic import BaseModel, Field


class MongoDBConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    mongo_uri: Optional[str] = Field(None, description="MongoDB connection URI")
    db_name: Optional[str] = Field("easy-memory", description="Database name")
