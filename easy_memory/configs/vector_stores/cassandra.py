from typing import Optional
from pydantic import BaseModel, Field

class CassandraConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    host: Optional[str] = Field(None, description="Cassandra host")
    port: Optional[int] = Field(None, description="Cassandra port")
    keyspace: Optional[str] = Field(None, description="Cassandra keyspace")