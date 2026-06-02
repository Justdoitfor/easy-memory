import json
import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class CassandraDB(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, host=None, port=None, keyspace=None):
        from cassandra.cluster import Cluster

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.cluster = Cluster([host or "127.0.0.1"], port=port or 9042)
        self.session = self.cluster.connect(keyspace or "mem0")
        self._create_table()

    def _create_table(self):
        self.session.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.collection_name} (
                id TEXT PRIMARY KEY,
                embedding LIST<FLOAT>,
                payload TEXT
            )"""
        )

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.embedding_model_dims = vector_size
        self._create_table()

    def insert(self, vectors, payloads=None, ids=None):
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            self.session.execute(
                f"INSERT INTO {self.collection_name} (id, embedding, payload) VALUES (%s, %s, %s)",
                (point_id, list(vec), json.dumps(payload)),
            )

    def search(self, query, vectors, top_k=5, filters=None):
        rows = self.session.execute(f"SELECT id, embedding, payload FROM {self.collection_name}")
        results = []
        for row in rows:
            embedding = row.embedding
            payload = json.loads(row.payload) if row.payload else {}
            dot = sum(a * b for a, b in zip(vectors, embedding))
            norm_a = sum(a * a for a in vectors) ** 0.5
            norm_b = sum(b * b for b in embedding) ** 0.5
            score = dot / (norm_a * norm_b) if norm_a and norm_b else 0
            results.append(
                type("Result", (), {"id": row.id, "score": score, "payload": payload})()
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete(self, vector_id):
        self.session.execute(
            f"DELETE FROM {self.collection_name} WHERE id = %s", (str(vector_id),)
        )

    def update(self, vector_id, vector=None, payload=None):
        if vector and payload:
            self.session.execute(
                f"UPDATE {self.collection_name} SET embedding=%s, payload=%s WHERE id=%s",
                (list(vector), json.dumps(payload), str(vector_id)),
            )
        elif payload:
            self.session.execute(
                f"UPDATE {self.collection_name} SET payload=%s WHERE id=%s",
                (json.dumps(payload), str(vector_id)),
            )

    def get(self, vector_id):
        row = self.session.execute(
            f"SELECT id, payload FROM {self.collection_name} WHERE id=%s",
            (str(vector_id),),
        ).one()
        if row:
            payload = json.loads(row.payload) if row.payload else {}
            return type("Result", (), {"id": row.id, "payload": payload})()
        return None

    def list_cols(self):
        rows = self.session.execute(
            "SELECT table_name FROM system_schema.tables WHERE keyspace_name=%s",
            (self.session.keyspace,),
        )
        return [r.table_name for r in rows]

    def delete_col(self):
        self.session.execute(f"DROP TABLE IF EXISTS {self.collection_name}")

    def col_info(self):
        row = self.session.execute(f"SELECT COUNT(*) as cnt FROM {self.collection_name}").one()
        return {"name": self.collection_name, "count": row.cnt if row else 0}

    def list(self, filters=None, top_k=100):
        rows = self.session.execute(f"SELECT id, payload FROM {self.collection_name} LIMIT %s", (top_k,))
        return [
            type(
                "Result",
                (),
                {
                    "id": r.id,
                    "payload": json.loads(r.payload) if r.payload else {},
                },
            )()
            for r in rows
        ]

    def reset(self):
        self.delete_col()
        self._create_table()
