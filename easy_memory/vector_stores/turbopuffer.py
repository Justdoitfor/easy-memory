import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class TurbopufferDB(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, api_key=None):
        import turbopuffer as tpuf

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.client = tpuf.Turbopuffer(api_key=api_key)
        self.ns = self.client.namespace(collection_name)

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.embedding_model_dims = vector_size
        self.ns = self.client.namespace(name)

    def insert(self, vectors, payloads=None, ids=None):
        id_list = [str(ids[i]) if ids else str(i) for i in range(len(vectors))]
        payload_list = payloads or [{}] * len(vectors)
        self.ns.upsert(
            ids=id_list,
            vectors=vectors,
            attributes=[{k: str(v) for k, v in p.items()} for p in payload_list],
        )

    def search(self, query, vectors, top_k=5, filters=None):
        kwargs = {"vector": vectors, "top_k": top_k}
        if filters:
            kwargs["filters"] = filters
        results = self.ns.query(**kwargs)
        return [
            type(
                "Result",
                (),
                {
                    "id": str(r.id),
                    "score": r.dist if hasattr(r, "dist") else 0,
                    "payload": r.attributes if hasattr(r, "attributes") else {},
                },
            )()
            for r in results
        ]

    def delete(self, vector_id):
        self.ns.delete(ids=[str(vector_id)])

    def update(self, vector_id, vector=None, payload=None):
        kwargs = {"ids": [str(vector_id)]}
        if vector:
            kwargs["vectors"] = [vector]
        if payload:
            kwargs["attributes"] = [{k: str(v) for k, v in payload.items()}]
        self.ns.upsert(**kwargs)

    def get(self, vector_id):
        results = self.ns.query(
            vector=[0.0] * self.embedding_model_dims,
            top_k=1,
            filters=["id", "==", str(vector_id)],
        )
        if results:
            r = results[0]
            return type(
                "Result",
                (),
                {
                    "id": str(r.id),
                    "payload": r.attributes if hasattr(r, "attributes") else {},
                },
            )()
        return None

    def list_cols(self):
        return [self.collection_name]

    def delete_col(self):
        self.ns.delete(filter=["id", "!=", ""])

    def col_info(self):
        return {"name": self.collection_name}

    def list(self, filters=None, top_k=100):
        kwargs = {"vector": [0.0] * self.embedding_model_dims, "top_k": top_k}
        if filters:
            kwargs["filters"] = filters
        results = self.ns.query(**kwargs)
        return [
            type(
                "Result",
                (),
                {
                    "id": str(r.id),
                    "payload": r.attributes if hasattr(r, "attributes") else {},
                },
            )()
            for r in results
        ]

    def reset(self):
        self.ns.delete(filter=["id", "!=", ""])
