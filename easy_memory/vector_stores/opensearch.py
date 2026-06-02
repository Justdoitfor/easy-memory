import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OpenSearchDB(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, host=None, port=None, api_key=None):
        from opensearchpy import OpenSearch

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.client = OpenSearch(
            hosts=[{"host": host or "localhost", "port": port or 9200}],
            http_auth=("admin", api_key) if api_key else None,
            use_ssl=False,
            verify_certs=False,
        )
        self._ensure_index()

    def _ensure_index(self):
        if not self.client.indices.exists(index=self.collection_name):
            mapping = {
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.embedding_model_dims,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                            },
                        },
                        "payload": {"type": "object", "enabled": True},
                    }
                },
            }
            self.client.indices.create(index=self.collection_name, body=mapping)

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.embedding_model_dims = vector_size
        self._ensure_index()

    def insert(self, vectors, payloads=None, ids=None):
        from opensearchpy.helpers import bulk

        actions = []
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            actions.append(
                {
                    "_index": self.collection_name,
                    "_id": point_id,
                    "_source": {"embedding": vec, "payload": payload},
                }
            )
        if actions:
            bulk(self.client, actions)

    def search(self, query, vectors, top_k=5, filters=None):
        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vectors,
                        "k": top_k,
                    }
                }
            },
        }
        if filters:
            body["query"]["knn"]["embedding"]["filter"] = {"term": filters}
        results = self.client.search(index=self.collection_name, body=body)
        return [
            type(
                "Result",
                (),
                {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "payload": hit["_source"].get("payload", {}),
                },
            )()
            for hit in results["hits"]["hits"]
        ]

    def delete(self, vector_id):
        self.client.delete(index=self.collection_name, id=str(vector_id))

    def update(self, vector_id, vector=None, payload=None):
        doc = {}
        if vector:
            doc["embedding"] = vector
        if payload:
            doc["payload"] = payload
        if doc:
            self.client.update(
                index=self.collection_name, id=str(vector_id), body={"doc": doc}
            )

    def get(self, vector_id):
        try:
            result = self.client.get(index=self.collection_name, id=str(vector_id))
            return type(
                "Result",
                (),
                {
                    "id": result["_id"],
                    "payload": result["_source"].get("payload", {}),
                    "vector": result["_source"].get("embedding"),
                },
            )()
        except Exception:
            return None

    def list_cols(self):
        indices = self.client.indices.get(index="*")
        return [name for name in indices.keys() if not name.startswith(".")]

    def delete_col(self):
        self.client.indices.delete(index=self.collection_name)

    def col_info(self):
        count = self.client.count(index=self.collection_name)["count"]
        return {"name": self.collection_name, "count": count}

    def list(self, filters=None, top_k=100):
        body = {"query": {"match_all": {}}, "size": top_k}
        results = self.client.search(index=self.collection_name, body=body)
        return [
            type(
                "Result",
                (),
                {
                    "id": hit["_id"],
                    "payload": hit["_source"].get("payload", {}),
                },
            )()
            for hit in results["hits"]["hits"]
        ]

    def reset(self):
        self.delete_col()
        self._ensure_index()
