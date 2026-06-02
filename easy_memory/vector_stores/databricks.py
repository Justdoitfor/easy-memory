import json
import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class Databricks(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, endpoint=None, token=None, catalog=None,
                 schema=None):
        from databricks import sql as dbsql

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.conn = dbsql.connect(
            server_hostname=endpoint,
            http_path=f"/sql/1.0/warehouses/{endpoint}",
            access_token=token,
        )
        self.full_table = f"{catalog or 'main'}.{schema or 'default'}.{collection_name}"
        self._create_table()

    def _create_table(self):
        with self.conn.cursor() as cur:
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.full_table} (
                    id STRING,
                    embedding ARRAY<FLOAT>,
                    payload STRING
                ) USING DELTA"""
            )

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.embedding_model_dims = vector_size
        self._create_table()

    def insert(self, vectors, payloads=None, ids=None):
        with self.conn.cursor() as cur:
            for i, vec in enumerate(vectors):
                point_id = str(ids[i]) if ids else str(i)
                payload = payloads[i] if payloads else {}
                cur.execute(
                    f"INSERT INTO {self.full_table} VALUES (%s, %s, %s)",
                    (point_id, list(vec), json.dumps(payload)),
                )

    def search(self, query, vectors, top_k=5, filters=None):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id, embedding, payload FROM {self.full_table}")
            rows = cur.fetchall()
            results = []
            for row in rows:
                embedding = row[1]
                payload = json.loads(row[2]) if row[2] else {}
                dot = sum(a * b for a, b in zip(vectors, embedding))
                norm_a = sum(a * a for a in vectors) ** 0.5
                norm_b = sum(b * b for b in embedding) ** 0.5
                score = dot / (norm_a * norm_b) if norm_a and norm_b else 0
                results.append(
                    type("Result", (), {"id": row[0], "score": score, "payload": payload})()
                )
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]

    def delete(self, vector_id):
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.full_table} WHERE id = %s", (str(vector_id),))

    def update(self, vector_id, vector=None, payload=None):
        with self.conn.cursor() as cur:
            if vector and payload:
                cur.execute(
                    f"UPDATE {self.full_table} SET embedding=%s, payload=%s WHERE id=%s",
                    (list(vector), json.dumps(payload), str(vector_id)),
                )
            elif payload:
                cur.execute(
                    f"UPDATE {self.full_table} SET payload=%s WHERE id=%s",
                    (json.dumps(payload), str(vector_id)),
                )

    def get(self, vector_id):
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, payload FROM {self.full_table} WHERE id=%s",
                (str(vector_id),),
            )
            row = cur.fetchone()
            if row:
                payload = json.loads(row[1]) if row[1] else {}
                return type("Result", (), {"id": row[0], "payload": payload})()
            return None

    def list_cols(self):
        with self.conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            return [r[1] for r in cur.fetchall()]

    def delete_col(self):
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.full_table}")

    def col_info(self):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.full_table}")
            return {"name": self.collection_name, "count": cur.fetchone()[0]}

    def list(self, filters=None, top_k=100):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id, payload FROM {self.full_table} LIMIT %s", (top_k,))
            return [
                type(
                    "Result",
                    (),
                    {
                        "id": r[0],
                        "payload": json.loads(r[1]) if r[1] else {},
                    },
                )()
                for r in cur.fetchall()
            ]

    def reset(self):
        self.delete_col()
        self._create_table()
