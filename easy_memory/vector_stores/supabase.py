import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class Supabase(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, supabase_url=None, supabase_key=None):
        from supabase import create_client

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.client = create_client(supabase_url, supabase_key)
        self.table_name = collection_name

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.table_name = name
        self.embedding_model_dims = vector_size

    def insert(self, vectors, payloads=None, ids=None):
        rows = []
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            rows.append({"id": point_id, "embedding": vec, "metadata": payload})
        if rows:
            self.client.table(self.table_name).upsert(rows).execute()

    def search(self, query, vectors, top_k=5, filters=None):
        result = self.client.rpc(
            "match_vectors",
            {
                "query_embedding": vectors,
                "match_count": top_k,
                "match_threshold": 0.0,
                "p_table_name": self.table_name,
            },
        ).execute()
        return [
            type(
                "Result",
                (),
                {
                    "id": row.get("id", ""),
                    "score": row.get("similarity", 0),
                    "payload": row.get("metadata", {}),
                },
            )()
            for row in (result.data or [])
        ]

    def delete(self, vector_id):
        self.client.table(self.table_name).delete().eq("id", str(vector_id)).execute()

    def update(self, vector_id, vector=None, payload=None):
        update = {}
        if vector:
            update["embedding"] = vector
        if payload:
            update["metadata"] = payload
        if update:
            self.client.table(self.table_name).update(update).eq("id", str(vector_id)).execute()

    def get(self, vector_id):
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("id", str(vector_id))
            .execute()
        )
        if result.data:
            row = result.data[0]
            return type(
                "Result",
                (),
                {
                    "id": row.get("id", ""),
                    "payload": row.get("metadata", {}),
                    "vector": row.get("embedding"),
                },
            )()
        return None

    def list_cols(self):
        result = self.client.rpc("list_tables").execute()
        return [row.get("table_name") for row in (result.data or [])]

    def delete_col(self):
        self.client.rpc("drop_table", {"table_name": self.table_name}).execute()

    def col_info(self):
        result = self.client.table(self.table_name).select("id", count="exact").execute()
        return {"name": self.collection_name, "count": result.count or 0}

    def list(self, filters=None, top_k=100):
        query = self.client.table(self.table_name).select("*").limit(top_k)
        result = query.execute()
        return [
            type(
                "Result",
                (),
                {
                    "id": row.get("id", ""),
                    "payload": row.get("metadata", {}),
                },
            )()
            for row in (result.data or [])
        ]

    def reset(self):
        self.client.table(self.table_name).delete().neq("id", "").execute()
