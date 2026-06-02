import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class UpstashVector(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", url=None, token=None):
        from upstash_vector import Index

        self.collection_name = collection_name
        self.index = Index(url=url, token=token)

    def create_col(self, name, vector_size, distance):
        self.collection_name = name

    def insert(self, vectors, payloads=None, ids=None):
        from upstash_vector import Vector

        items = []
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            items.append(Vector(id=point_id, vector=vec, metadata=payload))
        if items:
            self.index.upsert(vectors=items)

    def search(self, query, vectors, top_k=5, filters=None):
        kwargs = {"vector": vectors, "top_k": top_k}
        if filters:
            kwargs["filter"] = filters
        results = self.index.query(**kwargs)
        return [
            type(
                "Result",
                (),
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.metadata or {},
                },
            )()
            for r in results
        ]

    def delete(self, vector_id):
        self.index.delete(ids=[str(vector_id)])

    def update(self, vector_id, vector=None, payload=None):
        from upstash_vector import Vector

        kwargs = {"id": str(vector_id)}
        if vector:
            kwargs["vector"] = vector
        if payload:
            kwargs["metadata"] = payload
        self.index.upsert(vectors=[Vector(**kwargs)])

    def get(self, vector_id):
        results = self.index.fetch(ids=[str(vector_id)], include_metadata=True, include_vectors=True)
        if results:
            r = results[0]
            return type(
                "Result",
                (),
                {
                    "id": r.id,
                    "payload": r.metadata or {},
                    "vector": r.vector,
                },
            )()
        return None

    def list_cols(self):
        return [self.collection_name]

    def delete_col(self):
        self.index.reset()

    def col_info(self):
        info = self.index.info()
        return {"name": self.collection_name, "count": info.vector_count if hasattr(info, "vector_count") else 0}

    def list(self, filters=None, top_k=100):
        cursor = self.index.range(limit=top_k, include_metadata=True)
        return [
            type("Result", (), {"id": r.id, "payload": r.metadata or {}})()
            for r in cursor
        ]

    def reset(self):
        self.index.reset()
