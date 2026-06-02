import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class PGVector(VectorStoreBase):
    def __init__(
            self,
            collection_name="easy-memory",
            embedding_model_dims=1536,
            user=None,
            password=None,
            host=None,
            port=None,
            dbname=None,
    ):
        import psycopg

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.conn = psycopg.connect(conn_str)
        self._create_table()

    def _create_table(self):
        with self.conn.cursor() as cur:
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self.embedding_model_dims}),
                    payload JSONB
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
                    f"VALUES (%s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
                    f"embedding=EXCLUDED.embedding, payload=EXCLUDED.payload",
                    (point_id, vec, payload),
                )
            self.conn.commit()

    def search(self, query, vectors, top_k=5, filters=None):
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, embedding <=> %s::vector as distance, payload "
                f"FROM {self.collection_name} ORDER BY distance LIMIT %s",
                (vectors, top_k),
            )
            return [
                type("Result", (), {"id": r[0], "score": 1 - r[1], "payload": r[2]})()
                for r in cur.fetchall()
            ]

    def delete(self, vector_id):
        with self.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.collection_name} WHERE id = %s",
                (str(vector_id),),
            )
            self.conn.commit()

    def update(self, vector_id, vector=None, payload=None):
        with self.conn.cursor() as cur:
            if vector and payload:
                cur.execute(
                    f"UPDATE {self.collection_name} SET embedding=%s, payload=%s WHERE id=%s",
                    (vector, payload, str(vector_id)),
                )
            elif payload:
                cur.execute(
                    f"UPDATE {self.collection_name} SET payload=%s WHERE id=%s",
                    (payload, str(vector_id)),
                )
            self.conn.commit()

    def get(self, vector_id):
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, payload FROM {self.collection_name} WHERE id=%s",
                (str(vector_id),),
            )
            r = cur.fetchone()
            return (
                type("Result", (), {"id": r[0], "payload": r[1]})()
                if r
                else None
            )

    def list_cols(self):
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            )
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
            cur.execute(
                f"SELECT id, payload FROM {self.collection_name} LIMIT %s",
                (top_k,),
            )
            return [
                type("Result", (), {"id": r[0], "payload": r[1]})()
                for r in cur.fetchall()
            ]

    def reset(self):
        self.delete_col()
        self._create_table()
