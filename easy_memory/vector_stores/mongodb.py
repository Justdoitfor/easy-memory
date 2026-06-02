import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class MongoDB(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, mongo_uri=None, db_name="mem0"):
        from pymongo import MongoClient

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self._ensure_vector_index()

    def _ensure_vector_index(self):
        try:
            self.collection.create_index(
                [("embedding", "vectorSearch")],
                name="vector_index",
                vectorOptions={
                    "dimensions": self.embedding_model_dims,
                    "similarity": "cosine",
                },
            )
        except Exception as e:
            logger.debug(f"Vector index may already exist: {e}")

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.collection = self.db[name]
        self.embedding_model_dims = vector_size
        self._ensure_vector_index()

    def insert(self, vectors, payloads=None, ids=None):
        docs = []
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            payload = payloads[i] if payloads else {}
            doc = {"_id": point_id, "embedding": vec}
            doc.update(payload)
            docs.append(doc)
        if docs:
            from pymongo import ReplaceOne

            ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs]
            self.collection.bulk_write(ops)

    def search(self, query, vectors, top_k=5, filters=None):
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": vectors,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                }
            },
            {"$project": {"score": {"$meta": "vectorSearchScore"}, "embedding": 0}},
        ]
        if filters:
            pipeline.insert(1, {"$match": filters})
        results = list(self.collection.aggregate(pipeline))
        return [
            type(
                "Result",
                (),
                {
                    "id": str(r.get("_id", "")),
                    "score": r.get("score", 0.0),
                    "payload": {k: v for k, v in r.items() if k not in ("_id", "score")},
                },
            )()
            for r in results
        ]

    def delete(self, vector_id):
        self.collection.delete_one({"_id": str(vector_id)})

    def update(self, vector_id, vector=None, payload=None):
        update = {}
        if vector:
            update["embedding"] = vector
        if payload:
            update.update(payload)
        if update:
            self.collection.update_one({"_id": str(vector_id)}, {"$set": update})

    def get(self, vector_id):
        result = self.collection.find_one({"_id": str(vector_id)})
        if result:
            payload = {k: v for k, v in result.items() if k not in ("_id", "embedding")}
            return type(
                "Result",
                (),
                {
                    "id": str(result["_id"]),
                    "payload": payload,
                    "vector": result.get("embedding"),
                },
            )()
        return None

    def list_cols(self):
        return self.db.list_collection_names()

    def delete_col(self):
        self.collection.drop()

    def col_info(self):
        return {
            "name": self.collection_name,
            "count": self.collection.estimated_document_count(),
        }

    def list(self, filters=None, top_k=100):
        query = filters if filters else {}
        results = list(self.collection.find(query).limit(top_k))
        return [
            type(
                "Result",
                (),
                {
                    "id": str(r.get("_id", "")),
                    "payload": {k: v for k, v in r.items() if k not in ("_id", "embedding")},
                },
            )()
            for r in results
        ]

    def reset(self):
        self.delete_col()
        self.collection = self.db[self.collection_name]
        self._ensure_vector_index()
