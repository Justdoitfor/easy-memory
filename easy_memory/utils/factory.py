import importlib
from typing import Dict, Optional, Union

from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.configs.llms.anthropic import AnthropicConfig
from easy_memory.configs.llms.aws_bedrock import AWSBedrockConfig
from easy_memory.configs.llms.azure import AzureOpenAIConfig
from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.configs.llms.deepseek import DeepSeekConfig
from easy_memory.configs.llms.minimax import MinimaxConfig
from easy_memory.configs.llms.lmstudio import LMStudioConfig
from easy_memory.configs.llms.ollama import OllamaConfig
from easy_memory.configs.llms.openai import OpenAIConfig
from easy_memory.configs.llms.vllm import VllmConfig
from easy_memory.configs.rerankers.base import BaseRerankerConfig
from easy_memory.configs.rerankers.cohere import CohereRerankerConfig
from easy_memory.configs.rerankers.sentence_transformer import SentenceTransformerRerankerConfig
from easy_memory.configs.rerankers.zero_entropy import ZeroEntropyRerankerConfig
from easy_memory.configs.rerankers.llm import LLMRerankerConfig
from easy_memory.configs.rerankers.huggingface import HuggingFaceRerankerConfig
from easy_memory.embeddings.mock import MockEmbeddings


def load_class(class_type):
    """动态加载类的通用工具函数。

    接收形如 "easy_memory.llms.openai.OpenAILLM" 的全限定类名字符串，
    使用 importlib 动态导入模块并返回对应的类对象。

    工作流程：
    1. rsplit(".", 1) 将字符串拆分为模块路径和类名
    2. importlib.import_module() 动态导入模块
    3. getattr() 从模块中获取类对象

    Args:
        class_type: 全限定类名，格式为 "module.path.ClassName"

    Returns:
        对应的类对象（注意：是类本身，不是实例）
    """
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    """LLM 工厂类：根据 provider 名称创建对应的 LLM 实例。

    核心设计：
    - provider_to_class: 映射表，将 provider 名称映射到 (类路径, 配置类) 元组
    - create(): 工厂方法，负责实例化 LLM
    - register_provider(): 允许运行时注册新的 provider（插件化）
    """
    # 映射表：provider_name -> (类的全限定路径, 对应的配置类)
    provider_to_class = {
        "ollama": ("easy_memory.llms.ollama.OllamaLLM", OllamaConfig),
        "openai": ("easy_memory.llms.openai.OpenAILLM", OpenAIConfig),
        "groq": ("easy_memory.llms.groq.GroqLLM", BaseLlmConfig),
        "together": ("easy_memory.llms.together.TogetherLLM", BaseLlmConfig),
        "aws_bedrock": ("easy_memory.llms.aws_bedrock.AWSBedrockLLM", AWSBedrockConfig),
        "litellm": ("easy_memory.llms.litellm.LiteLLM", BaseLlmConfig),
        "azure_openai": ("easy_memory.llms.azure_openai.AzureOpenAILLM", AzureOpenAIConfig),
        "openai_structured": ("easy_memory.llms.openai_structured.OpenAIStructuredLLM", OpenAIConfig),
        "anthropic": ("easy_memory.llms.anthropic.AnthropicLLM", AnthropicConfig),
        "azure_openai_structured": ("easy_memory.llms.azure_openai_structured.AzureOpenAIStructuredLLM",
                                    AzureOpenAIConfig),
        "gemini": ("easy_memory.llms.gemini.GeminiLLM", BaseLlmConfig),
        "deepseek": ("easy_memory.llms.deepseek.DeepSeekLLM", DeepSeekConfig),
        "minimax": ("easy_memory.llms.minimax.MiniMaxLLM", MinimaxConfig),
        "xai": ("easy_memory.llms.xai.XAILLM", BaseLlmConfig),
        "sarvam": ("easy_memory.llms.sarvam.SarvamLLM", BaseLlmConfig),
        "lmstudio": ("easy_memory.llms.lmstudio.LMStudioLLM", LMStudioConfig),
        "vllm": ("easy_memory.llms.vllm.VllmLLM", VllmConfig),
        "langchain": ("easy_memory.llms.langchain.LangchainLLM", BaseLlmConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseLlmConfig, Dict]] = None, **kwargs):
        """工厂方法：根据 provider 名称创建 LLM 实例。

        处理三种 config 输入形式：
        1. None —— 用默认参数创建配置
        2. dict —— 用字典构造配置
        3. BaseLlmConfig 实例 —— 可能需要升级为更具体的配置子类

        Args:
            provider_name: provider 名称，如 "openai", "ollama"
            config: 配置对象或字典，可选
            **kwargs: 额外配置参数

        Returns:
            LLM 实例

        Raises:
            ValueError: 不支持的 provider
        """
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")

        # 从映射表获取类路径和配置类
        class_type, config_class = cls.provider_to_class[provider_name]
        llm_class = load_class(class_type)

        # 处理配置：统一转换为目标 provider 需要的配置类型
        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config.update(kwargs)
            config = config_class(**config)
        elif isinstance(config, BaseLlmConfig):
            # 如果传入的是基础配置但 provider 需要更具体的配置（如 OpenAIConfig），
            # 需要将基础字段复制到具体配置中
            if config_class != BaseLlmConfig:
                config_dict = {
                    "model": config.model, "temperature": config.temperature,
                    "api_key": config.api_key, "max_tokens": config.max_tokens,
                    "top_p": config.top_p, "top_k": config.top_k,
                    "enable_vision": config.enable_vision, "vision_details": config.vision_details,
                    "http_client_proxies": config.http_client,
                }
                config_dict.update(kwargs)
                config = config_class(**config_dict)

        return llm_class(config)

    @classmethod
    def register_provider(cls, name: str, class_path: str, config_class=None):
        """运行时注册新的 provider（插件化机制）。

        允许第三方代码在不修改工厂源码的情况下注册自定义 LLM。
        """
        if config_class is None:
            config_class = BaseLlmConfig
        cls.provider_to_class[name] = (class_path, config_class)

    @classmethod
    def get_supported_providers(cls) -> list:
        """返回所有支持的 provider 名称列表。"""
        return list(cls.provider_to_class.keys())


class EmbedderFactory:
    """Embedding 工厂类：根据 provider 名称创建 Embedding 实例。

    与 LlmFactory 类似，但 EmbedderFactory 更简单——
    所有 embedding provider 使用相同的 BaseEmbedderConfig。
    """
    provider_to_class = {
        "openai": "easy_memory.embeddings.openai.OpenAIEmbedding",
        "ollama": "easy_memory.embeddings.ollama.OllamaEmbedding",
        "huggingface": "easy_memory.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "easy_memory.embeddings.azure_openai.AzureOpenAIEmbedding",
        "gemini": "easy_memory.embeddings.gemini.GoogleGenAIEmbedding",
        "vertexai": "easy_memory.embeddings.vertexai.VertexAIEmbedding",
        "together": "easy_memory.embeddings.together.TogetherEmbedding",
        "lmstudio": "easy_memory.embeddings.lmstudio.LMStudioEmbedding",
        "langchain": "easy_memory.embeddings.langchain.LangchainEmbedding",
        "aws_bedrock": "easy_memory.embeddings.aws_bedrock.AWSBedrockEmbedding",
        "fastembed": "easy_memory.embeddings.fastembed.FastEmbedEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config, vector_config: Optional[dict]):
        """工厂方法：创建 Embedding 实例。

        特殊逻辑：如果 vector store 自带 embedding（如 upstash_vector），
        则使用 MockEmbeddings 占位，实际嵌入由 vector store 完成。
        """
        if provider_name == "upstash_vector" and vector_config and vector_config.enable_embeddings:
            return MockEmbeddings()
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    """向量存储工厂类：根据 provider 名称创建向量存储实例。

    与前两个工厂不同，VectorStoreFactory 的 create 方法
    接受字典配置并直接展开为关键字参数传给构造函数。
    """
    provider_to_class = {
        "qdrant": "easy_memory.vector_stores.qdrant.Qdrant",
        "chroma": "easy_memory.vector_stores.chroma.ChromaDB",
        "pgvector": "easy_memory.vector_stores.pgvector.PGVector",
        "milvus": "easy_memory.vector_stores.milvus.MilvusDB",
        "upstash_vector": "easy_memory.vector_stores.upstash_vector.UpstashVector",
        "azure_ai_search": "easy_memory.vector_stores.azure_ai_search.AzureAISearch",
        "azure_mysql": "easy_memory.vector_stores.azure_mysql.AzureMySQL",
        "pinecone": "easy_memory.vector_stores.pinecone.PineconeDB",
        "mongodb": "easy_memory.vector_stores.mongodb.MongoDB",
        "redis": "easy_memory.vector_stores.redis.RedisDB",
        "valkey": "easy_memory.vector_stores.valkey.ValkeyDB",
        "databricks": "easy_memory.vector_stores.databricks.Databricks",
        "elasticsearch": "easy_memory.vector_stores.elasticsearch.ElasticsearchDB",
        "vertex_ai_vector_search": "easy_memory.vector_stores.vertex_ai_vector_search.GoogleMatchingEngine",
        "opensearch": "easy_memory.vector_stores.opensearch.OpenSearchDB",
        "supabase": "easy_memory.vector_stores.supabase.Supabase",
        "weaviate": "easy_memory.vector_stores.weaviate.Weaviate",
        "faiss": "easy_memory.vector_stores.faiss.FAISS",
        "langchain": "easy_memory.vector_stores.langchain.Langchain",
        "s3_vectors": "easy_memory.vector_stores.s3_vectors.S3Vectors",
        "baidu": "easy_memory.vector_stores.baidu.BaiduDB",
        "cassandra": "easy_memory.vector_stores.cassandra.CassandraDB",
        "neptune": "easy_memory.vector_stores.neptune_analytics.NeptuneAnalyticsVector",
        "turbopuffer": "easy_memory.vector_stores.turbopuffer.TurbopufferDB",
    }

    @classmethod
    def create(cls, provider_name, config):
        """工厂方法：创建向量存储实例。

        如果 config 是 Pydantic 模型，先用 model_dump() 转为字典，
        再展开为关键字参数传给构造函数。
        """
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

    @classmethod
    def reset(cls, instance):
        """重置向量存储实例（清空所有数据并重建集合）。"""
        instance.reset()
        return instance


class RerankerFactory:
    """Reranker 工厂类：根据 provider 名称创建重排序器实例。

    Reranker 用于对检索结果进行二次排序，提升相关性。
    结构与 LlmFactory 类似，支持运行时插件注册。
    """
    provider_to_class = {
        "cohere": ("easy_memory.reranker.cohere_reranker.CohereReranker", CohereRerankerConfig),
        "sentence_transformer": ("easy_memory.reranker.sentence_transformer_reranker.SentenceTransformerReranker",
                                 SentenceTransformerRerankerConfig),
        "zero_entropy": ("easy_memory.reranker.zero_entropy_reranker.ZeroEntropyReranker", ZeroEntropyRerankerConfig),
        "llm_reranker": ("easy_memory.reranker.llm_reranker.LLMReranker", LLMRerankerConfig),
        "huggingface": ("easy_memory.reranker.huggingface_reranker.HuggingFaceReranker", HuggingFaceRerankerConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseRerankerConfig, Dict]] = None, **kwargs):
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported reranker provider: {provider_name}")

        class_path, config_class = cls.provider_to_class[provider_name]

        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config = config_class(**config, **kwargs)
        elif not isinstance(config, BaseRerankerConfig):
            raise ValueError(f"Config must be a {config_class.__name__} instance or dict")

        try:
            reranker_class = load_class(class_path)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import reranker for provider '{provider_name}': {e}")

        return reranker_class(config)
