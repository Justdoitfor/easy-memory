import json
import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class ValkeyDB(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, redis_url=None):
        import numpy as np
        from redis import Redis
        from redis.commands.search.field import NumericField, TagField, VectorField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        self.np = np
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.client = Redis.from_url(redis_url)
        self.index_name = f"idx:{collection_name}"
        self._create_index()

    def _create_index(self):
        from redis.commands.search.field import TagField, VectorField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        try:
            self.client.ft(self.index_name).info()
        except Exception:
            schema = (
                TagField("$.id", as_name="id"),
                VectorField(
                    "$.embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embedding_model_dims,
                        "DISTANCE_METRIC": "COSINE",
                    },
                    as_name="embedding",
                ),
            )
            self.client.ft(self.index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[f"{self.collection_name}:"], index_type=IndexType.JSON
                ),
            )

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.index_name = f"idx:{name}"
        self.embedding_model_dims = vector_size
        self._create_index()

    def _doc_key(self, doc_id):
        return f"{self.collection_name}:{doc_id}"

    def insert(self, vectors, payloads=None, ids=None):
        pipe = self.client.pipeline()
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            doc = {
                "id": point_id,
                "embedding": self.np.array(vec, dtype=self.np.float32).tobytes().hex(),
                **payload,
            }
            pipe.json().set(self._doc_key(point_id), "$", doc)
        pipe.execute()

    def search(self, query, vectors, top_k=5, filters=None):
        from redis.commands.search.query import Query

        query_vec = self.np.array(vectors, dtype=self.np.float32).tobytes()
        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("id", "score")
            .dialect(2)
        )
        results = self.client.ft(self.index_name).search(
            q, query_params={"vec": query_vec}
        )
        return [
            type(
                "Result",
                (),
                {
                    "id": getattr(doc, "id", ""),
                    "score": float(getattr(doc, "score", 0)),
                    "payload": json.loads(
                        self.client.json().get(self._doc_key(getattr(doc, "id", ""))) or "{}"
                    ),
                },
            )()
            for doc in results.docs
        ]

    def delete(self, vector_id):
        self.client.delete(self._doc_key(str(vector_id)))

    def update(self, vector_id, vector=None, payload=None):
        key = self._doc_key(str(vector_id))
        if vector:
            self.client.json().set(
                key, "$.embedding", self.np.array(vector, dtype=self.np.float32).tobytes().hex()
            )
        if payload:
            for k, v in payload.items():
                self.client.json().set(key, f"$.{k}", v)

    def get(self, vector_id):
        key = self._doc_key(str(vector_id))
        doc = self.client.json().get(key)
        if doc is None:
            return None
        payload = {k: v for k, v in doc.items() if k not in ("id", "embedding")}
        return type("Result", (), {"id": doc.get("id", ""), "payload": payload})()

    def list_cols(self):
        return [name.decode() for name in self.client.execute_command("FT._LIST")]

    def delete_col(self):
        self.client.ft(self.index_name).dropindex(delete_documents=True)

    def col_info(self):
        info = self.client.ft(self.index_name).info()
        return {"name": self.collection_name, "count": info.get("num_docs", "0")}

    def list(self, filters=None, top_k=100):
        from redis.commands.search.query import Query

        q = Query("*").paging(0, top_k)
        results = self.client.ft(self.index_name).search(q)
        out = []
        for doc in results.docs:
            key = self._doc_key(getattr(doc, "id", ""))
            full = self.client.json().get(key) or {}
            payload = {k: v for k, v in full.items() if k not in ("id", "embedding")}
            out.append(type("Result", (), {"id": full.get("id", ""), "payload": payload})())
        return out

    def reset(self):
        self.delete_col()
        self._create_index()