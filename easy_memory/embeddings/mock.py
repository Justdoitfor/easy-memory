from typing import Literal, Optional

from easy_memory.embeddings.base import EmbeddingBase


class MockEmbeddings(EmbeddingBase):
    """Mock 嵌入实现：返回固定的假向量。

    用途：
    1. 测试：不需要真实 API 调用就能测试向量存储等下游组件
    2. 占位：当向量存储自带 embedding 功能时（如 upstash_vector），
       使用 MockEmbeddings 避免重复嵌入
    """

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """返回固定的 10 维向量。"""
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
