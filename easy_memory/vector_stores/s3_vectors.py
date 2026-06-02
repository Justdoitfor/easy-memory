import json
import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class S3Vectors(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, bucket=None, region=None):
        import boto3

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.bucket = bucket
        self.s3 = boto3.client("s3", region_name=region)
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except Exception:
            try:
                self.s3.create_bucket(Bucket=self.bucket)
            except Exception as e:
                logger.warning(f"Could not create bucket: {e}")

    def _key(self, vector_id):
        return f"{self.collection_name}/{vector_id}.json"

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.embedding_model_dims = vector_size

    def insert(self, vectors, payloads=None, ids=None):
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            doc = {"id": point_id, "embedding": vec, "payload": payload}
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._key(point_id),
                Body=json.dumps(doc),
                ContentType="application/json",
            )

    def search(self, query, vectors, top_k=5, filters=None):
        paginator = self.s3.get_paginator("list_objects_v2")
        results = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.collection_name}/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("__meta__.json"):
                    continue
                body = self.s3.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                doc = json.loads(body)
                embedding = doc.get("embedding", [])
                dot = sum(a * b for a, b in zip(vectors, embedding))
                norm_a = sum(a * a for a in vectors) ** 0.5
                norm_b = sum(b * b for b in embedding) ** 0.5
                score = dot / (norm_a * norm_b) if norm_a and norm_b else 0
                results.append(
                    type(
                        "Result",
                        (),
                        {
                            "id": doc.get("id", ""),
                            "score": score,
                            "payload": doc.get("payload", {}),
                        },
                    )()
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete(self, vector_id):
        self.s3.delete_object(Bucket=self.bucket, Key=self._key(str(vector_id)))

    def update(self, vector_id, vector=None, payload=None):
        existing = self.get(vector_id)
        if existing:
            doc = {
                "id": str(vector_id),
                "embedding": vector or getattr(existing, "vector", []),
                "payload": payload or existing.payload,
            }
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._key(str(vector_id)),
                Body=json.dumps(doc),
                ContentType="application/json",
            )

    def get(self, vector_id):
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=self._key(str(vector_id)))["Body"].read()
            doc = json.loads(body)
            return type(
                "Result",
                (),
                {
                    "id": doc.get("id", ""),
                    "payload": doc.get("payload", {}),
                    "vector": doc.get("embedding"),
                },
            )()
        except Exception:
            return None

    def list_cols(self):
        paginator = self.s3.get_paginator("list_objects_v2")
        prefixes = set()
        for page in paginator.paginate(Bucket=self.bucket, Delimiter="/"):
            for prefix in page.get("CommonPrefixes", []):
                prefixes.add(prefix["Prefix"].rstrip("/"))
        return list(prefixes)

    def delete_col(self):
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.collection_name}/"):
            for obj in page.get("Contents", []):
                self.s3.delete_object(Bucket=self.bucket, Key=obj["Key"])

    def col_info(self):
        paginator = self.s3.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.collection_name}/"):
            count += len([o for o in page.get("Contents", []) if not o["Key"].endswith("__meta__.json")])
        return {"name": self.collection_name, "count": count}

    def list(self, filters=None, top_k=100):
        paginator = self.s3.get_paginator("list_objects_v2")
        results = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.collection_name}/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("__meta__.json"):
                    continue
                if len(results) >= top_k:
                    break
                body = self.s3.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                doc = json.loads(body)
                results.append(
                    type("Result", (), {"id": doc.get("id", ""), "payload": doc.get("payload", {})})()
                )
        return results

    def reset(self):
        self.delete_col()
