import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class Langchain(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, vector_store=None):
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        if vector_store is None:
            raise ValueError("A LangChain vector_store instance must be provided")
        self.vector_store = vector_store

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.embedding_model_dims = vector_size

    def insert(self, vectors, payloads=None, ids=None):
        from langchain_core.documents import Document

        documents = []
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            documents.append(Document(page_content=payload.get("data", ""), metadata=payload, id=point_id))
        if documents:
            self.vector_store.add_documents(documents)

    def search(self, query, vectors, top_k=5, filters=None):
        results = self.vector_store.similarity_search_by_vector(vectors, k=top_k, filter=filters)
        return [
            type(
                "Result",
                (),
                {
                    "id": getattr(doc, "id", str(i)),
                    "score": 0.0,
                    "payload": doc.metadata,
                },
            )()
            for i, doc in enumerate(results)
        ]

    def delete(self, vector_id):
        self.vector_store.delete(ids=[str(vector_id)])

    def update(self, vector_id, vector=None, payload=None):
        from langchain_core.documents import Document

        doc = Document(page_content=payload.get("data", "") if payload else "", metadata=payload or {},
                       id=str(vector_id))
        self.vector_store.add_documents([doc])

    def get(self, vector_id):
        results = self.vector_store.get_by_ids([str(vector_id)])
        if results:
            doc = results[0]
            return type(
                "Result",
                (),
                {
                    "id": getattr(doc, "id", str(vector_id)),
                    "payload": doc.metadata,
                    "vector": None,
                },
            )()
        return None

    def list_cols(self):
        return [self.collection_name]

    def delete_col(self):
        self.vector_store.delete(where={})

    def col_info(self):
        return {"name": self.collection_name}

    def list(self, filters=None, top_k=100):
        results = self.vector_store.similarity_search("", k=top_k, filter=filters)
        return [
            type("Result", (), {"id": getattr(doc, "id", str(i)), "payload": doc.metadata})()
            for i, doc in enumerate(results)
        ]

    def reset(self):
        self.delete_col()
