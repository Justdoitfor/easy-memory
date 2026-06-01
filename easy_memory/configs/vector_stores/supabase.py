from typing import Optional
from pydantic import BaseModel, Field


class SupabaseConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    supabase_url: Optional[str] = Field(None, description="Supabase project URL")
    supabase_key: Optional[str] = Field(None, description="Supabase API key")
