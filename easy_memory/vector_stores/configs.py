""" 使用 `model_validator` 实现了动态配置加载：根据 `provider` 字符串，动态导入对应的配置类并实例化 """
from typing import Dict, Optional
from pydantic import BaseModel, Field, model_validator


class VectorStoreConfig(BaseModel):
    provider: str = Field(description="Provider of the vector store (e.g., 'qdrant', 'chroma', 'upstash_vector')",
                          default="qdrant")
    config: Optional[Dict] = Field(description="Configuration for the specific vector store",
                                   default=None)
    _provider_configs: Dict[str, str] = {
        "qdrant": "QdrantConfig",
        "chroma": "ChromaDbConfig",
        "pgvector": "PGVectorConfig",
        "pinecone": "PineconeConfig",
        "mongodb": "MongoDBConfig",
        "milvus": "MilvusDBConfig",
        "baidu": "BaiduDBConfig",
        "cassandra": "CassandraConfig",
        "neptune": "NeptuneAnalyticsConfig",
        "upstash_vector": "UpstashVectorConfig",
        "azure_ai_search": "AzureAISearchConfig",
        "azure_mysql": "AzureMySQLConfig",
        "redis": "RedisDBConfig",
        "valkey": "ValkeyConfig",
        "databricks": "DatabricksConfig",
        "elasticsearch": "ElasticsearchConfig",
        "vertex_ai_vector_search": "GoogleMatchingEngineConfig",
        "opensearch": "OpenSearchConfig",
        "supabase": "SupabaseConfig",
        "weaviate": "WeaviateConfig",
        "faiss": "FAISSConfig",
        "langchain": "LangchainConfig",
        "s3_vectors": "S3VectorsConfig",
        "turbopuffer": "TurbopufferConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "VectorStoreConfig":
        provider = self.provider
        config = self.config
        if provider not in self._provider_configs:
            raise ValueError(f"Unsupported vector store provider: {provider}")

        module = __import__(
            f"easy_memory.configs.vector_stores.{provider}",
            fromlist=[self._provider_configs[provider]]
        )
        config_class = getattr(module, self._provider_configs[provider])
        if config is None:
            config = {}
        if not isinstance(config, dict):
            if not isinstance(config, config_class):
                raise ValueError(f"Invalid config type for provider: {provider}")
            return self
        if "path" not in config and "path" in config_class.__annotations__:
            config["path"] = f"/tmp/{provider}"
        self.config = config_class(**config)
        return self
