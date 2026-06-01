import os
import warnings
from typing import Literal, Optional

from openai import OpenAI

from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class OpenAIEmbedding(EmbeddingBase):
    """OpenAI Embedding 实现。

    将文本转换为向量（浮点数列表），用于语义搜索。

    核心概念：
    - 单条 embed(text)：将一个文本转为向量
    - 批量 embed_batch(texts)：将多个文本转为向量（更高效）
    - 维度（dimensions）：向量的长度，如 1536 或 3072
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)
        # 默认使用 text-embedding-3-small 模型（性价比最高）
        self.config.model = self.config.model or "text-embedding-3-small"
        # 如果用户指定了维度，需要传给 API（OpenAI 支持自定义维度）
        self._pass_dimensions_to_api = self.config.embedding_dims is not None
        # 默认维度 1536（text-embedding-3-small 的默认输出维度）
        self.config.embedding_dims = self.config.embedding_dims or 1536

        # 初始化 OpenAI 客户端
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
                self.config.openai_base_url
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
        )
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated. "
                "Please use 'OPENAI_BASE_URL' instead.",
                DeprecationWarning,
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """将单个文本转换为向量。

        Args:
            text: 要嵌入的文本
            memory_action: 操作类型（某些模型可能针对不同场景优化）

        Returns:
            list[float]: 嵌入向量
        """
        text = text.replace("\n", " ")
        kwargs = {"input": [text], "model": self.config.model, "encoding_format": "float"}
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self.config.embedding_dims
        return self.client.embeddings.create(**kwargs).data[0].embedding

    def embed_batch(self, texts, memory_action="add"):
        """批量将文本转换为向量。

        分批处理以避免超过 API 限制（单次最多 100 条）。
        按 index 排序确保返回顺序与输入一致。

        Args:
            texts: 文本列表
            memory_action: 操作类型

        Returns:
            list[list[float]]: 嵌入向量列表
        """
        MAX_BATCH = 100
        texts = [text.replace("\n", " ") for text in texts]
        all_embeddings = []
        for i in range(0, len(texts), MAX_BATCH):
            chunk = texts[i: i + MAX_BATCH]
            kwargs = {"input": chunk, "model": self.config.model, "encoding_format": "float"}
            if self._pass_dimensions_to_api:
                kwargs["dimensions"] = self.config.embedding_dims
            response = self.client.embeddings.create(**kwargs)
            all_embeddings.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return all_embeddings
