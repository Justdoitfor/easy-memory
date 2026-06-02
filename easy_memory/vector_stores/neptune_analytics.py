import json
import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class NeptuneAnalyticsVector(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, host=None, port=None):
        import boto3

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.graph_id = collection_name
        self.host = host or "localhost"
        self.port = port or 8182
        self._ensure_graph()

    def _ensure_graph(self):
        logger.info(f"Neptune Analytics graph '{self.graph_id}' assumed to exist externally")

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.graph_id = name
        self.embedding_model_dims = vector_size

    def insert(self, vectors, payloads=None, ids=None):
        import requests

        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            query = f"""
            CREATE (n:Vector {{id: $id, payload: $payload}})
            WITH n CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
            YIELD node RETURN node
            """
            requests.post(
                f"http://{self.host}:{self.port}/openCypher",
                json={
                    "query": query,
                    "parameters": {
                        "id": point_id,
                        "payload": json.dumps(payload),
                        "embedding": vec,
                    },
                },
            )

    def search(self, query, vectors, top_k=5, filters=None):
        import requests

        cypher = f"""
        CALL db.index.vector.queryNodes('embedding', {top_k}, $queryVector)
        YIELD node, score
        RETURN node.id AS id, node.payload AS payload, score
        """
        resp = requests.post(
            f"http://{self.host}:{self.port}/openCypher",
            json={"query": cypher, "parameters": {"queryVector": vectors}},
        )
        data = resp.json()
        results = []
        for record in data.get("results", []):
            payload = {}
            try:
                payload = json.loads(record.get("payload", "{}"))
            except (json.JSONDecodeError, TypeError):
                payload = {}
            results.append(
                type(
                    "Result",
                    (),
                    {
                        "id": record.get("id", ""),
                        "score": record.get("score", 0),
                        "payload": payload,
                    },
                )()
            )
        return results

    def delete(self, vector_id):
        import requests

        requests.post(
            f"http://{self.host}:{self.port}/openCypher",
            json={
                "query": "MATCH (n:Vector {id: $id}) DETACH DELETE n",
                "parameters": {"id": str(vector_id)},
            },
        )

    def update(self, vector_id, vector=None, payload=None):
        import requests

        if payload:
            requests.post(
                f"http://{self.host}:{self.port}/openCypher",
                json={
                    "query": "MATCH (n:Vector {id: $id}) SET n.payload = $payload",
                    "parameters": {"id": str(vector_id), "payload": json.dumps(payload)},
                },
            )

    def get(self, vector_id):
        import requests

        resp = requests.post(
            f"http://{self.host}:{self.port}/openCypher",
            json={
                "query": "MATCH (n:Vector {id: $id}) RETURN n.id AS id, n.payload AS payload",
                "parameters": {"id": str(vector_id)},
            },
        )
        data = resp.json()
        results = data.get("results", [])
        if results:
            record = results[0]
            payload = {}
            try:
                payload = json.loads(record.get("payload", "{}"))
            except (json.JSONDecodeError, TypeError):
                payload = {}
            return type("Result", (), {"id": record.get("id", ""), "payload": payload})()
        return None

    def list_cols(self):
        return [self.collection_name]

    def delete_col(self):
        import requests

        requests.post(
            f"http://{self.host}:{self.port}/openCypher",
            json={"query": "MATCH (n:Vector) DETACH DELETE n"},
        )

    def col_info(self):
        import requests

        resp = requests.post(
            f"http://{self.host}:{self.port}/openCypher",
            json={"query": "MATCH (n:Vector) RETURN count(n) AS cnt"},
        )
        data = resp.json()
        cnt = data.get("results", [{}])[0].get("cnt", 0) if data.get("results") else 0
        return {"name": self.collection_name, "count": cnt}

    def list(self, filters=None, top_k=100):
        import requests

        resp = requests.post(
            f"http://{self.host}:{self.port}/openCypher",
            json={
                "query": f"MATCH (n:Vector) RETURN n.id AS id, n.payload AS payload LIMIT {top_k}",
            },
        )
        data = resp.json()
        results = []
        for record in data.get("results", []):
            payload = {}
            try:
                payload = json.loads(record.get("payload", "{}"))
            except (json.JSONDecodeError, TypeError):
                payload = {}
            results.append(
                type("Result", (), {"id": record.get("id", ""), "payload": payload})()
            )
        return results

    def reset(self):
        self.delete_col()
