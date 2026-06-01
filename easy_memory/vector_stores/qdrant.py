import logging
import re
from typing import Optional

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchExcept,
    MatchText,
    MatchValue,
    PointIdsList,
    PointStruct,
    PointVectors,
    Range,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class Qdrant(VectorStoreBase):
    """Qdrant 向量数据库实现。

    Qdrant 是一个高性能的向量搜索引擎，支持：
    - 密集向量搜索（语义搜索）
    - 稀疏向量搜索（BM25 关键词搜索）
    - 混合搜索（同时使用语义和关键词）
    - 结构化过滤（按字段条件过滤结果）

    连接方式：
    - 本地模式：path 参数指定本地存储路径
    - 远程模式：host/port 或 url 参数连接远程服务
    """

    def __init__(
            self,
            collection_name: str,
            embedding_model_dims: int,
            client: QdrantClient = None,
            host: str = None,
            port: int = None,
            path: str = None,
            url: str = None,
            api_key: str = None,
            on_disk: bool = False,
    ):
        # 初始化客户端：支持传入已有客户端或自动创建
        if client:
            self.client = client
            self.is_local = False
        else:
            params = {}
            if api_key:
                params["api_key"] = api_key
            if url:
                params["url"] = url
            if host and port:
                params["host"] = host
                params["port"] = port
            if not params:
                params["path"] = path
                self.is_local = True
            else:
                self.is_local = False
            self.client = QdrantClient(**params)

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.on_disk = on_disk
        # BM25 编码器（懒加载）
        self._bm25_encoder = None
        self._has_bm25_slot = False
        # 自动创建集合
        self.create_col(embedding_model_dims, on_disk)

    def _get_bm25_encoder(self):
        """懒加载 BM25 稀疏向量编码器。

        BM25 (Best Matching 25) 是经典的关键词检索算法。
        它将文本转换为稀疏向量——只有少数维度有非零值，
        表示文本中每个词的重要性。

        需要安装 fastembed: pip install fastembed
        """
        if self._bm25_encoder is None:
            try:
                from fastembed import SparseTextEmbedding
                self._bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
                logger.info("BM25 encoder loaded (fastembed Qdrant/bm25)")
            except ImportError:
                logger.warning("fastembed not installed — BM25 keyword search disabled.")
                self._bm25_encoder = False
            except Exception as e:
                logger.warning(f"Failed to load BM25 encoder: {e}")
                self._bm25_encoder = False
        return self._bm25_encoder if self._bm25_encoder is not False else None

    def _encode_bm25(self, text: str):
        """将文本编码为 BM25 稀疏向量。

        稀疏向量只存储非零值，格式：
        - indices: 非零值的位置索引
        - values: 对应的值
        """
        encoder = self._get_bm25_encoder()
        if encoder is None:
            return None
        try:
            results = list(encoder.embed([text]))
            if results:
                sparse = results[0]
                return SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())
        except Exception as e:
            logger.debug(f"BM25 encoding failed: {e}")
        return None

    def create_col(self, vector_size: int, on_disk: bool, distance: Distance = Distance.COSINE):
        """创建或验证集合。

        如果集合已存在，检查是否支持 BM25 稀疏向量；
        如果不存在，创建新集合（同时配置密集向量和稀疏向量）。
        """
        response = self.list_cols()
        for collection in response.collections:
            if collection.name == self.collection_name:
                logger.debug(f"Collection {self.collection_name} already exists. Skipping creation.")
                info = self.client.get_collection(self.collection_name)
                sparse_cfg = info.config.params.sparse_vectors
                self._has_bm25_slot = bool(sparse_cfg and "bm25" in sparse_cfg)
                if not self._has_bm25_slot:
                    logger.warning(
                        f"Collection '{self.collection_name}' predates v3 hybrid search (no 'bm25' sparse slot). "
                        "BM25 keyword scoring will be disabled for this collection."
                    )
                self._create_filter_indexes()
                return

        # 创建新集合：同时配置密集向量和 BM25 稀疏向量
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance, on_disk=on_disk),
            sparse_vectors_config={"bm25": SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        self._has_bm25_slot = True
        self._create_filter_indexes()

    def _create_filter_indexes(self):
        """为常用过滤字段创建索引。

        索引可以显著加速按字段过滤的查询。
        仅对远程连接创建（本地模式 Qdrant 自动处理）。
        """
        if self.is_local:
            return
        common_fields = ["user_id", "agent_id", "run_id", "actor_id"]
        for field in common_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name, field_name=field, field_schema="keyword"
                )
            except Exception as e:
                logger.debug(f"Index for {field} might already exist: {e}")

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        """插入向量数据。

        每个点（point）包含：
        - id: 唯一标识
        - vector: 向量数据（密集向量 + 可选的 BM25 稀疏向量）
        - payload: 附加数据（如原始文本、元数据）

        混合搜索：同时存储密集向量（语义）和稀疏向量（关键词），
        搜索时可以融合两种检索方式的结果。
        """
        points = []
        for idx, vector in enumerate(vectors):
            payload = payloads[idx] if payloads else {}
            point_id = idx if ids is None else ids[idx]
            named_vectors = {"": vector}  # 空字符串键 = 默认密集向量
            if self._has_bm25_slot:
                # 从 payload 中提取文本用于 BM25 编码
                text_for_bm25 = payload.get("text_lemmatized") or payload.get("data", "")
                if text_for_bm25:
                    sparse = self._encode_bm25(text_for_bm25)
                    if sparse is not None:
                        named_vectors["bm25"] = sparse
            points.append(PointStruct(id=point_id, vector=named_vectors, payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points)

    # ISO 日期时间正则表达式，用于判断范围过滤是否为日期类型
    _ISO_DATETIME_RE = re.compile(
        r"^\d{4}-\d{2}-\d{2}" r"([T ]\d{2}:\d{2}(:\d{2})?" r"(\.\d+)?" r"(Z|[+-]\d{2}:?\d{2})?" r")?$"
    )

    @staticmethod
    def _is_datetime_range(range_kwargs: dict) -> bool:
        """判断范围参数是否为日期时间类型。"""
        return all(isinstance(v, str) and Qdrant._ISO_DATETIME_RE.match(v) for v in range_kwargs.values())

    def _build_field_condition(self, key: str, value) -> Optional[FieldCondition]:
        """将单个字段过滤条件转换为 Qdrant 的 FieldCondition。

        支持的过滤语法：
        - 精确匹配: {"field": value}
        - 列表匹配: {"field": [v1, v2, ...]} -> MatchAny
        - 通配符: {"field": "*"} -> 跳过
        - 比较运算: {"field": {"gt": 5, "lt": 10}} -> Range
        - 相等: {"field": {"eq": value}}
        - 不等: {"field": {"ne": value}}
        - 包含: {"field": {"in": [v1, v2]}}
        - 不包含: {"field": {"nin": [v1, v2]}}
        - 文本匹配: {"field": {"contains": "text"}}
        """
        if not isinstance(value, dict):
            if value == "*":
                return None
            if isinstance(value, list):
                return FieldCondition(key=key, match=MatchAny(any=value))
            return FieldCondition(key=key, match=MatchValue(value=value))

        ops = set(value.keys())
        range_ops = {"gt", "gte", "lt", "lte"}
        non_range_ops = ops - range_ops

        if ops & range_ops:
            if non_range_ops:
                raise ValueError(f"Cannot mix range operators with non-range operators for field '{key}'.")
            range_kwargs = {op: value[op] for op in range_ops if op in value}
            if self._is_datetime_range(range_kwargs):
                try:
                    return FieldCondition(key=key, range=DatetimeRange(**range_kwargs))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid datetime value in range filter for field '{key}': {e}") from e
            return FieldCondition(key=key, range=Range(**range_kwargs))
        elif "eq" in value:
            return FieldCondition(key=key, match=MatchValue(value=value["eq"]))
        elif "ne" in value:
            return FieldCondition(key=key, match=MatchExcept(**{"except": [value["ne"]]}))
        elif "in" in value:
            return FieldCondition(key=key, match=MatchAny(any=value["in"]))
        elif "nin" in value:
            return FieldCondition(key=key, match=MatchExcept(**{"except": value["nin"]}))
        elif "contains" in value or "icontains" in value:
            op = "icontains" if "icontains" in value else "contains"
            text = value[op]
            return FieldCondition(key=key, match=MatchText(text=text))
        else:
            supported = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"}
            raise ValueError(f"Unsupported filter operator(s) for field '{key}': {ops}. Supported: {supported}")

    def _create_filter(self, filters: dict) -> Optional[Filter]:
        """将字典形式的过滤条件转换为 Qdrant 的 Filter 对象。

        支持的逻辑运算：
        - $and / AND: 所有条件都满足
        - $or / OR: 任一条件满足
        - $not / NOT: 所有条件都不满足

        示例：
        {
            "AND": [
                {"user_id": "user1"},
                {"OR": [{"category": "news"}, {"category": "blog"}]}
            ]
        }
        """
        if not filters:
            return None
        key_map = {"$or": "OR", "$not": "NOT", "$and": "AND"}
        normalized = {}
        for key, value in filters.items():
            norm_key = key_map.get(key, key)
            if norm_key not in normalized:
                normalized[norm_key] = value

        must, should, must_not = [], [], []
        for key, value in normalized.items():
            if key in ("AND", "OR", "NOT"):
                if not isinstance(value, list):
                    raise ValueError(f"{key} filter value must be a list, got {type(value).__name__}")
                for item in value:
                    if not isinstance(item, dict):
                        raise ValueError(f"{key} filter list item must be a dict, got {type(item).__name__}")

            if key == "AND":
                for sub in value:
                    built = self._create_filter(sub)
                    if built:
                        must.append(built)
            elif key == "OR":
                for sub in value:
                    built = self._create_filter(sub)
                    if built:
                        should.append(built)
            elif key == "NOT":
                for sub in value:
                    built = self._create_filter(sub)
                    if built:
                        must_not.append(built)
            else:
                condition = self._build_field_condition(key, value)
                if condition is not None:
                    must.append(condition)

        if not any([must, should, must_not]):
            return None
        return Filter(must=must or None, should=should or None, must_not=must_not or None)

    def search(self, query: str, vectors: list, top_k: int = 5, filters: dict = None) -> list:
        """向量搜索：返回与查询向量最相似的 top_k 个结果。

        Args:
            query: 查询文本（用于日志）
            vectors: 查询向量
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            搜索结果列表
        """
        query_filter = self._create_filter(filters) if filters else None
        hits = self.client.query_points(
            collection_name=self.collection_name, query=vectors, query_filter=query_filter, limit=top_k,
        )
        return hits.points

    def search_batch(self, queries: list, vectors_list: list, top_k: int = 1, filters: dict = None):
        """批量向量搜索：同时执行多个查询。

        优先使用批量 API，失败则回退到逐条查询。
        """
        query_filter = self._create_filter(filters) if filters else None
        requests = [
            models.QueryRequest(query=vec, filter=query_filter, limit=top_k, with_payload=True)
            for vec in vectors_list
        ]
        try:
            results = self.client.query_batch_points(collection_name=self.collection_name, requests=requests)
            return [r.points for r in results]
        except Exception as e:
            logger.warning(f"Batch search failed, falling back to sequential: {e}")
            return [self.search(q, v, top_k=top_k, filters=filters) for q, v in zip(queries, vectors_list)]

    def keyword_search(self, query, top_k=5, filters=None):
        """BM25 关键词搜索：使用稀疏向量进行关键词匹配。

        与语义搜索互补：
        - 语义搜索：理解"意思相近"的文本
        - 关键词搜索：精确匹配关键词
        - 混合搜索：融合两者的优势
        """
        if not self._has_bm25_slot:
            return None
        sparse_query = self._encode_bm25(query)
        if sparse_query is None:
            return None
        try:
            query_filter = self._create_filter(filters) if filters else None
            hits = self.client.query_points(
                collection_name=self.collection_name, query=sparse_query, using="bm25",
                query_filter=query_filter, limit=top_k,
            )
            return hits.points
        except Exception as e:
            logger.debug(f"BM25 keyword search failed: {e}")
            return None

    def delete(self, vector_id: int):
        """删除指定 ID 的向量。"""
        self.client.delete(collection_name=self.collection_name, points_selector=PointIdsList(points=[vector_id]))

    def update(self, vector_id: int, vector: list = None, payload: dict = None):
        """更新向量和/或 payload。

        三种更新模式：
        1. 同时更新向量和 payload -> upsert
        2. 仅更新 payload -> set_payload
        3. 仅更新向量 -> update_vectors
        """
        if vector is not None and payload is not None:
            named_vectors = {"": vector}
            if self._has_bm25_slot:
                text_for_bm25 = payload.get("text_lemmatized") or payload.get("data", "")
                if text_for_bm25:
                    sparse = self._encode_bm25(text_for_bm25)
                    if sparse is not None:
                        named_vectors["bm25"] = sparse
            point = PointStruct(id=vector_id, vector=named_vectors, payload=payload)
            self.client.upsert(collection_name=self.collection_name, points=[point])
        else:
            if payload is not None:
                self.client.set_payload(collection_name=self.collection_name, payload=payload, points=[vector_id])
            if vector is not None:
                self.client.update_vectors(
                    collection_name=self.collection_name, points=[PointVectors(id=vector_id, vector=vector)]
                )

    def get(self, vector_id: int) -> dict:
        """根据 ID 获取向量及其 payload。"""
        result = self.client.retrieve(collection_name=self.collection_name, ids=[vector_id], with_payload=True)
        return result[0] if result else None

    def list_cols(self) -> list:
        """列出所有集合。"""
        return self.client.get_collections()

    def delete_col(self):
        """删除当前集合。"""
        self.client.delete_collection(collection_name=self.collection_name)

    def col_info(self) -> dict:
        """获取当前集合信息。"""
        return self.client.get_collection(collection_name=self.collection_name)

    def list(self, filters: dict = None, top_k: int = 100) -> list:
        """列出集合中的向量（带过滤）。"""
        query_filter = self._create_filter(filters) if filters else None
        return self.client.scroll(
            collection_name=self.collection_name, scroll_filter=query_filter,
            limit=top_k, with_payload=True, with_vectors=False,
        )

    def reset(self):
        """重置集合：删除并重建。"""
        self.delete_col()
        self.create_col(self.embedding_model_dims, self.on_disk)
