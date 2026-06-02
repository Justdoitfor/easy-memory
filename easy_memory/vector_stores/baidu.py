import json
import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class BaiduDB(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, host=None, port=None, user=None,
                 password=None):
        from pymochow import MochowClient
        from pymochow.model.table import Schema, Table

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.client = MochowClient(endpoint=f"http://{host or 'localhost'}:{port or 5287}", user=user,
                                   password=password)
        self.db = self.client.database("mem0_db")
        self._ensure_table()

    def _ensure_table(self):
        from pymochow.model.table import Field, FieldType, Schema, Table, VectorIndex

        try:
            self.table = self.db.describe_table(self.collection_name)
        except Exception:
            fields = [
                Field("id", FieldType.STRING, primary_key=True),
                Field("embedding", FieldType.FLOAT_VECTOR, dim=self.embedding_model_dims),
                Field("payload", FieldType.STRING),
            ]
            schema = Schema(fields=fields, primary_key="id")
            self.db.create_table(self.collection_name, schema=schema)
            self.table = self.db.describe_table(self.collection_name)

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.embedding_model_dims = vector_size
        self._ensure_table()

    def insert(self, vectors, payloads=None, ids=None):
        rows = []
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            rows.append({"id": point_id, "embedding": vec, "payload": json.dumps(payload)})
        if rows:
            self.table.upsert(rows)

    def search(self, query, vectors, top_k=5, filters=None):
        results = self.table.vector_search(
            field="embedding", vector=vectors, topk=top_k
        )
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
                        "id": r.get("id", ""),
                        "score": r.get("score", 0),
                        "payload": payload,
                    },
                )()
            )
        return out

    def delete(self, vector_id):
        self.table.delete({"id": str(vector_id)})

    def update(self, vector_id, vector=None, payload=None):
        row = {"id": str(vector_id)}
        if vector:
            row["embedding"] = vector
        if payload:
            row["payload"] = json.dumps(payload)
        self.table.upsert([row])

    def get(self, vector_id):
        result = self.table.get({"id": str(vector_id)})
        if result:
            payload = {}
            try:
                payload = json.loads(result.get("payload", "{}"))
            except (json.JSONDecodeError, TypeError):
                payload = {}
            return type(
                "Result",
                (),
                {
                    "id": result.get("id", ""),
                    "payload": payload,
                    "vector": result.get("embedding"),
                },
            )()
        return None

    def list_cols(self):
        return self.db.list_tables()

    def delete_col(self):
        self.db.drop_table(self.collection_name)

    def col_info(self):
        return {"name": self.collection_name}

    def list(self, filters=None, top_k=100):
        results = self.table.scan(limit=top_k)
        out = []
        for r in results:
            payload = {}
            try:
                payload = json.loads(r.get("payload", "{}"))
            except (json.JSONDecodeError, TypeError):
                payload = {}
            out.append(type("Result", (), {"id": r.get("id", ""), "payload": payload})())
        return out

    def reset(self):
        self.delete_col()
        self._ensure_table()
