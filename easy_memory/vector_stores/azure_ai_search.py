import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class AzureAISearch(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, api_key=None, endpoint=None):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes import SearchIndexClient

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.endpoint = endpoint
        self.credential = AzureKeyCredential(api_key)
        self.index_client = SearchIndexClient(endpoint=endpoint, credential=self.credential)
        self._ensure_index()

    def _ensure_index(self):
        from azure.search.documents.indexes.models import (
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchAlgorithmConfiguration,
            VectorSearchProfile,
        )

        try:
            self.index_client.get_index(self.collection_name)
        except Exception:
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self.embedding_model_dims,
                    vector_search_profile_name="default",
                ),
                SearchField(name="payload", type=SearchFieldDataType.String, searchable=False),
            ]
            vector_search = VectorSearch(
                profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="default")],
                algorithms=[VectorSearchAlgorithmConfiguration(name="default", kind="hnsw")],
            )
            index = SearchIndex(name=self.collection_name, fields=fields, vector_search=vector_search)
            self.index_client.create_or_update_index(index)
        self.search_client = self.index_client.get_search_client(self.collection_name)

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.embedding_model_dims = vector_size
        self._ensure_index()

    def insert(self, vectors, payloads=None, ids=None):
        import json

        docs = []
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            docs.append({"id": point_id, "embedding": vec, "payload": json.dumps(payload)})
        if docs:
            self.search_client.upload_documents(docs)

    def search(self, query, vectors, top_k=5, filters=None):
        import json

        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(vector=vectors, k_nearest_neighbors=top_k, fields="embedding")
        results = self.search_client.search(vector_queries=[vector_query], top=top_k)
        out = []
        for r in results:
            payload = {}
            try:
                payload = json.loads(r.get("payload", "{}"))
            except (json.JSONDecodeError, TypeError):
                payload = {}
            out.append(
                type(
                    "Result",
                    (),
                    {
                        "id": r["id"],
                        "score": r.get("@search.score", 0),
                        "payload": payload,
                    },
                )()
            )
        return out

    def delete(self, vector_id):
        self.search_client.delete_documents([{"id": str(vector_id)}])

    def update(self, vector_id, vector=None, payload=None):
        import json

        doc = {"id": str(vector_id)}
        if vector:
            doc["embedding"] = vector
        if payload:
            doc["payload"] = json.dumps(payload)
        self.search_client.merge_or_upload_documents([doc])

    def get(self, vector_id):
        import json

        try:
            result = self.search_client.get_document(key=str(vector_id))
            payload = {}
            try:
                payload = json.loads(result.get("payload", "{}"))
            except (json.JSONDecodeError, TypeError):
                payload = {}
            return type(
                "Result",
                (),
                {
                    "id": result["id"],
                    "payload": payload,
                    "vector": result.get("embedding"),
                },
            )()
        except Exception:
            return None

    def list_cols(self):
        return [idx.name for idx in self.index_client.list_indexes()]

    def delete_col(self):
        self.index_client.delete_index(self.collection_name)

    def col_info(self):
        stats = self.search_client.get_document_count()
        return {"name": self.collection_name, "count": stats}

    def list(self, filters=None, top_k=100):
        import json

        results = self.search_client.search(search_text="*", top=top_k)
        out = []
        for r in results:
            payload = {}
            try:
                payload = json.loads(r.get("payload", "{}"))
            except (json.JSONDecodeError, TypeError):
                payload = {}
            out.append(type("Result", (), {"id": r["id"], "payload": payload})())
        return out

    def reset(self):
        self.delete_col()
        self._ensure_index()
