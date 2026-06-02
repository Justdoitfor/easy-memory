import logging

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class GoogleMatchingEngine(VectorStoreBase):
    def __init__(
            self,
            collection_name="easy-memory",
            embedding_model_dims=1536,
            project_id=None,
            location=None,
            index_id=None,
            endpoint_id=None,
    ):
        from google.cloud import aiplatform

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.project_id = project_id
        self.location = location
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        aiplatform.init(project=project_id, location=location)
        self.index = aiplatform.MatchingEngineIndex(index_name=index_id)
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id)

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.embedding_model_dims = vector_size

    def insert(self, vectors, payloads=None, ids=None):
        from google.cloud import aiplatform

        datapoints = []
        for i, vec in enumerate(vectors):
            point_id = str(ids[i]) if ids else str(i)
            datapoints.append(
                aiplatform.matching_engine.matching_engine_index.Datapoint(
                    datapoint_id=point_id,
                    feature_vector=vec,
                    restricts=[],
                )
            )
        if datapoints:
            self.index.upsert_datapoints(datapoints=datapoints)

    def search(self, query, vectors, top_k=5, filters=None):
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.index_id,
            queries=[vectors],
            num_neighbors=top_k,
        )
        results = []
        for neighbor in response[0]:
            results.append(
                type(
                    "Result",
                    (),
                    {
                        "id": neighbor.id,
                        "score": neighbor.distance if hasattr(neighbor, "distance") else 0,
                        "payload": {},
                    },
                )()
            )
        return results

    def delete(self, vector_id):
        self.index.remove_datapoints(datapoint_ids=[str(vector_id)])

    def update(self, vector_id, vector=None, payload=None):
        from google.cloud import aiplatform

        if vector:
            dp = aiplatform.matching_engine.matching_engine_index.Datapoint(
                datapoint_id=str(vector_id),
                feature_vector=vector,
            )
            self.index.upsert_datapoints(datapoints=[dp])

    def get(self, vector_id):
        logger.warning("GoogleMatchingEngine.get() is not directly supported by the Vertex AI API")
        return None

    def list_cols(self):
        from google.cloud import aiplatform

        indexes = aiplatform.MatchingEngineIndex.list(
            project=self.project_id, location=self.location
        )
        return [idx.resource_name for idx in indexes]

    def delete_col(self):
        self.index.delete()

    def col_info(self):
        return {
            "name": self.collection_name,
            "index_id": self.index_id,
            "endpoint_id": self.endpoint_id,
        }

    def list(self, filters=None, top_k=100):
        logger.warning("GoogleMatchingEngine.list() is not directly supported by the Vertex AI API")
        return []

    def reset(self):
        logger.warning(
            "GoogleMatchingEngine.reset() is not directly supported; please recreate the index manually"
        )
