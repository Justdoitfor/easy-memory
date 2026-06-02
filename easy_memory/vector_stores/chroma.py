import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class ChromaDB(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", hosting_url=None, path="/tmp/chroma_db"):
        import chromadb

        if hosting_url:
            self.client = chromadb.HttpClient(host=hosting_url)
        else:
            self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def create_col(self, name, vector_size, distance):
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def insert(self, vectors, payloads=None, ids=None):
        self.collection.add(
            ids=[str(i) for i in ids] if ids else [str(i) for i in range(len(vectors))],
            embeddings=vectors,
            metadatas=payloads,
        )

    def search(self, query, vectors, top_k=5, filters=None):
        where = filters if filters else None
        results = self.collection.query(
            query_embeddings=[vectors], n_results=top_k, where=where
        )
        return [
            type(
                "Result",
                (),
                {
                    "id": results["ids"][0][i],
                    "score": 1 - results["distances"][0][i],
                    "payload": results["metadatas"][0][i] if results["metadatas"] else {},
                },
            )()
            for i in range(len(results["ids"][0]))
        ]

    def delete(self, vector_id):
        self.collection.delete(ids=[str(vector_id)])

    def update(self, vector_id, vector=None, payload=None):
        kwargs = {"ids": [str(vector_id)]}
        if vector:
            kwargs["embeddings"] = [vector]
        if payload:
            kwargs["metadatas"] = [payload]
        self.collection.update(**kwargs)

    def get(self, vector_id):
        result = self.collection.get(
            ids=[str(vector_id)], include=["embeddings", "metadatas"]
        )
        if not result["ids"]:
            return None
        return type(
            "Result",
            (),
            {
                "id": result["ids"][0],
                "payload": result["metadatas"][0] if result["metadatas"] else {},
                "vector": result["embeddings"][0] if result["embeddings"] else None,
            },
        )()

    def list_cols(self):
        return self.client.list_collections()

    def delete_col(self):
        self.client.delete_collection(self.collection.name)

    def col_info(self):
        return {"name": self.collection.name, "count": self.collection.count()}

    def list(self, filters=None, top_k=100):
        where = filters if filters else None
        result = self.collection.get(
            where=where, limit=top_k, include=["embeddings", "metadatas"]
        )
        return [
            type(
                "Result",
                (),
                {
                    "id": result["ids"][i],
                    "payload": result["metadatas"][i] if result["metadatas"] else {},
                    "vector": result["embeddings"][i] if result["embeddings"] else None,
                },
            )()
            for i in range(len(result["ids"]))
        ]

    def reset(self):
        self.delete_col()
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name, metadata={"hnsw:space": "cosine"}
        )
