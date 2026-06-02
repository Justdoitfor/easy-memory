import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class MilvusDB(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", embedding_model_dims=1536, url=None, token=None):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        connections.connect(alias="default", uri=url, token=token)
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_model_dims),
                FieldSchema(name="payload", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields)
            self.collection = Collection(collection_name, schema)
            self.collection.create_index(
                "vector",
                {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}},
            )
        else:
            self.collection = Collection(collection_name)
        self.collection.load()

    def create_col(self, name, vector_size, distance):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if not utility.has_collection(name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
                FieldSchema(name="payload", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields)
            self.collection = Collection(name, schema)
            self.collection.create_index(
                "vector",
                {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}},
            )
        else:
            self.collection = Collection(name)
        self.collection.load()

    def insert(self, vectors, payloads=None, ids=None):
        data = [str(ids[i]) if ids else str(i) for i in range(len(vectors))]
        self.collection.insert([data, vectors, payloads or [{}] * len(vectors)])
        self.collection.flush()

    def search(self, query, vectors, top_k=5, filters=None):
        results = self.collection.search(
            data=[vectors],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["payload"],
        )
        return [
            type(
                "Result",
                (),
                {"id": r.id, "score": r.distance, "payload": r.entity.get("payload")},
            )()
            for r in results[0]
        ]

    def delete(self, vector_id):
        self.collection.delete(f'id == "{str(vector_id)}"')

    def update(self, vector_id, vector=None, payload=None):
        self.delete(vector_id)
        self.insert(
            [vector] if vector else [[]],
            [payload] if payload else [{}],
            [str(vector_id)],
        )

    def get(self, vector_id):
        results = self.collection.query(
            expr=f'id == "{str(vector_id)}"', output_fields=["payload"]
        )
        return (
            type(
                "Result",
                (),
                {"id": str(vector_id), "payload": results[0].get("payload")},
            )()
            if results
            else None
        )

    def list_cols(self):
        from pymilvus import utility

        return utility.list_collections()

    def delete_col(self):
        from pymilvus import utility

        utility.drop_collection(self.collection_name)

    def col_info(self):
        return {"name": self.collection_name, "count": self.collection.num_entities}

    def list(self, filters=None, top_k=100):
        results = self.collection.query(
            expr="id != ''", output_fields=["payload"], limit=top_k
        )
        return [
            type("Result", (), {"id": r["id"], "payload": r.get("payload")})()
            for r in results
        ]

    def reset(self):
        self.delete_col()
        self.__init__(self.collection_name, self.embedding_model_dims)
