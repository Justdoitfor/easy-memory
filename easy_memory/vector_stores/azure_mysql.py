import json
import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class AzureMySQL(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, host=None, port=None, user=None,
                 password=None, database=None):
        import mysql.connector

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.conn = mysql.connector.connect(
            host=host or "localhost",
            port=port or 3306,
            user=user,
            password=password,
            database=database,
        )
        self._create_table()

    def _create_table(self):
        with self.conn.cursor() as cur:
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id VARCHAR(255) PRIMARY KEY,
                    embedding JSON,
                    payload JSON
                )"""
            )
            self.conn.commit()

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
                    f"INSERT INTO {self.collection_name} (id, embedding, payload) "
                    f"VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE embedding=VALUES(embedding), payload=VALUES(payload)",
                    (point_id, json.dumps(vec), json.dumps(payload)),
                )
            self.conn.commit()

    def search(self, query, vectors, top_k=5, filters=None):
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, embedding, payload FROM {self.collection_name} LIMIT %s",
                (top_k,),
            )
            rows = cur.fetchall()
            results = []
            for row in rows:
                embedding = json.loads(row[1]) if isinstance(row[1], str) else row[1]
                payload = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                # Compute cosine similarity
                dot = sum(a * b for a, b in zip(vectors, embedding))
                norm_a = sum(a * a for a in vectors) ** 0.5
                norm_b = sum(b * b for b in embedding) ** 0.5
                score = dot / (norm_a * norm_b) if norm_a and norm_b else 0
                results.append(
                    type("Result", (), {"id": row[0], "score": score, "payload": payload or {}})()
                )
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]

    def delete(self, vector_id):
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.collection_name} WHERE id = %s", (str(vector_id),))
            self.conn.commit()

    def update(self, vector_id, vector=None, payload=None):
        with self.conn.cursor() as cur:
            if vector and payload:
                cur.execute(
                    f"UPDATE {self.collection_name} SET embedding=%s, payload=%s WHERE id=%s",
                    (json.dumps(vector), json.dumps(payload), str(vector_id)),
                )
            elif payload:
                cur.execute(
                    f"UPDATE {self.collection_name} SET payload=%s WHERE id=%s",
                    (json.dumps(payload), str(vector_id)),
                )
            self.conn.commit()

    def get(self, vector_id):
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, embedding, payload FROM {self.collection_name} WHERE id=%s",
                (str(vector_id),),
            )
            r = cur.fetchone()
            if not r:
                return None
            payload = json.loads(r[2]) if isinstance(r[2], str) else r[2]
            return type("Result", (), {"id": r[0], "payload": payload or {}})()

    def list_cols(self):
        with self.conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            return [r[0] for r in cur.fetchall()]

    def delete_col(self):
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
            self.conn.commit()

    def col_info(self):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
            return {"name": self.collection_name, "count": cur.fetchone()[0]}

    def list(self, filters=None, top_k=100):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id, payload FROM {self.collection_name} LIMIT %s", (top_k,))
            return [
                type(
                    "Result",
                    (),
                    {
                        "id": r[0],
                        "payload": json.loads(r[1]) if isinstance(r[1], str) else (r[1] or {}),
                    },
                )()
                for r in cur.fetchall()
            ]

    def reset(self):
        self.delete_col()
        self._create_table()
