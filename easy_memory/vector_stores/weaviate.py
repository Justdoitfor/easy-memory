import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class Weaviate(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, host=None, port=None, api_key=None):
        import weaviate

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        url = f"http://{host or 'localhost'}:{port or 8080}"
        if api_key:
            self.client = weaviate.connect_to_custom(
                http_host=host or "localhost",
                http_port=port or 8080,
                http_secure=False,
                auth_credentials=weaviate.auth.AuthApiKey(api_key),
            )
        else:
            self.client = weaviate.connect_to_local(
                http_host=host or "localhost",
                http_port=port or 8080,
            )
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=None,  # manual vectors
                properties=[
                    {"name": "payload", "dataType": ["text"]},
                ],
            )

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self._ensure_collection()

    def insert(self, vectors, payloads=None, ids=None):
        import uuid

        collection = self.client.collections.get(self.collection_name)
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            collection.data.insert(
                properties={"payload": str(payload)},
                vector=vec,
                uuid=uuid.uuid5(uuid.NAMESPACE_DNS, point_id),
            )

    def search(self, query, vectors, top_k=5, filters=None):
        import json

        collection = self.client.collections.get(self.collection_name)
        response = collection.query.near_vector(
            near_vector=vectors,
            limit=top_k,
            return_metadata=["distance"],
            return_properties=["payload"],
        )
        results = []
        for obj in response.objects:
            payload = {}
            try:
                payload = json.loads(obj.properties.get("payload", "{}"))
            except (json.JSONDecodeError, TypeError):
                payload = {}
            results.append(
                type(
                    "Result",
                    (),
                    {
                        "id": str(obj.uuid),
                        "score": 1 - obj.metadata.distance if obj.metadata.distance else 0,
                        "payload": payload,
                    },
                )()
            )
        return results

    def delete(self, vector_id):
        import uuid

        collection = self.client.collections.get(self.collection_name)
        collection.data.delete_by_id(uuid.uuid5(uuid.NAMESPACE_DNS, str(vector_id)))

    def update(self, vector_id, vector=None, payload=None):
        import uuid

        uid = uuid.uuid5(uuid.NAMESPACE_DNS, str(vector_id))
        collection = self.client.collections.get(self.collection_name)
        kwargs = {}
        if payload:
            kwargs["properties"] = {"payload": str(payload)}
        if vector:
            kwargs["vector"] = vector
        collection.data.update(uuid=uid, **kwargs)

    def get(self, vector_id):
        import json
        import uuid

        uid = uuid.uuid5(uuid.NAMESPACE_DNS, str(vector_id))
        collection = self.client.collections.get(self.collection_name)
        obj = collection.query.fetch_object_by_id(uid)
        if obj is None:
            return None
        payload = {}
        try:
            payload = json.loads(obj.properties.get("payload", "{}"))
        except (json.JSONDecodeError, TypeError):
            payload = {}
        return type("Result", (), {"id": str(obj.uuid), "payload": payload})()

    def list_cols(self):
        return [c.name for c in self.client.collections.list_all()]

    def delete_col(self):
        self.client.collections.delete(self.collection_name)

    def col_info(self):
        collection = self.client.collections.get(self.collection_name)
        count = len(collection.query.fetch_objects(limit=10000).objects)
        return {"name": self.collection_name, "count": count}

    def list(self, filters=None, top_k=100):
        import json

        collection = self.client.collections.get(self.collection_name)
        response = collection.query.fetch_objects(
            limit=top_k, return_properties=["payload"]
        )
        results = []
        for obj in response.objects:
            payload = {}
            try:
                payload = json.loads(obj.properties.get("payload", "{}"))
            except (json.JSONDecodeError, TypeError):
                payload = {}
            results.append(
                type("Result", (), {"id": str(obj.uuid), "payload": payload})()
            )
        return results

    def reset(self):
        self.delete_col()
        self._ensure_collection()
