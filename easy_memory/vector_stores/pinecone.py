import logging
import os

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class PineconeDB(VectorStoreBase):
    def __init__(
            self,
            collection_name="easy-memory",
            api_key=None,
            environment=None,
            server_url=None,
            metric="cosine",
            batch_size=100,
    ):
        from pinecone import Pinecone

        self.collection_name = collection_name
        self.metric = metric
        self.batch_size = batch_size
        self.pc = Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))
        existing = [idx.name for idx in self.pc.list_indexes()]
        if collection_name not in existing:
            self.pc.create_index(name=collection_name, dimension=1536, metric=metric)
        self.index = self.pc.Index(collection_name)

    def create_col(self, name, vector_size, distance):
        existing = [idx.name for idx in self.pc.list_indexes()]
        if name not in existing:
            self.pc.create_index(name=name, dimension=vector_size, metric=self.metric)
        self.index = self.pc.Index(name)

    def insert(self, vectors, payloads=None, ids=None):
        batch = []
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            batch.append({"id": point_id, "values": vec, "metadata": payload})
            if len(batch) >= self.batch_size:
                self.index.upsert(vectors=batch)
                batch = []
        if batch:
            self.index.upsert(vectors=batch)

    def search(self, query, vectors, top_k=5, filters=None):
        results = self.index.query(
            vector=vectors,
            top_k=top_k,
            filter=filters,
            include_metadata=True,
        )
        return [
            type(
                "Result",
                (),
                {
                    "id": m["id"],
                    "score": m["score"],
                    "payload": m.get("metadata", {}),
                },
            )()
            for m in results["matches"]
        ]

    def delete(self, vector_id):
        self.index.delete(ids=[str(vector_id)])

    def update(self, vector_id, vector=None, payload=None):
        self.insert(
            [vector] if vector else [[]],
            [payload] if payload else [{}],
            [str(vector_id)],
        )

    def get(self, vector_id):
        result = self.index.fetch(ids=[str(vector_id)])
        if vector_id in result["vectors"]:
            v = result["vectors"][vector_id]
            return type("Result", (), {"id": vector_id, "payload": v.get("metadata", {})})()
        return None

    def list_cols(self):
        return [idx.name for idx in self.pc.list_indexes()]

    def delete_col(self):
        self.pc.delete_index(self.collection_name)

    def col_info(self):
        return self.index.describe_index_stats()

    def list(self, filters=None, top_k=100):
        results = self.index.query(
            vector=[0.0] * 1536,
            top_k=top_k,
            filter=filters,
            include_metadata=True,
        )
        return [
            type("Result", (), {"id": m["id"], "payload": m.get("metadata", {})})()
            for m in results["matches"]
        ]

    def reset(self):
        self.delete_col()
        self.__init__(self.collection_name, metric=self.metric)
