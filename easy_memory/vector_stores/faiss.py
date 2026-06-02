import json
import logging
import os

from easy_memory.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class FAISS(VectorStoreBase):
    def __init__(self, collection_name="easy-memory", path=None):
        import faiss
        import numpy as np

        self.faiss = faiss
        self.np = np
        self.collection_name = collection_name
        self.path = path or "/tmp/faiss"
        self.index_path = os.path.join(self.path, f"{collection_name}.index")
        self.meta_path = os.path.join(self.path, f"{collection_name}.json")
        os.makedirs(self.path, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = self.faiss.read_index(self.index_path)
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.index = None
            self.metadata = {}  # id -> payload

    def _save(self):
        if self.index is not None:
            self.faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w") as f:
                json.dump(self.metadata, f)

    def create_col(self, name, vector_size, distance):
        self.collection_name = name
        self.index = self.faiss.IndexFlatIP(vector_size)
        self.metadata = {}
        self._save()

    def insert(self, vectors, payloads=None, ids=None):
        import numpy as np

        vecs = np.array(vectors, dtype=np.float32)
        if self.index is None:
            self.index = self.faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        base = len(self.metadata)
        for i in range(len(vectors)):
            point_id = str(ids[i]) if ids else str(base + i)
            payload = payloads[i] if payloads else {}
            self.metadata[point_id] = payload
        self._save()

    def search(self, query, vectors, top_k=5, filters=None):
        import numpy as np

        if self.index is None or self.index.ntotal == 0:
            return []
        vec = np.array([vectors], dtype=np.float32)
        scores, indices = self.index.search(vec, min(top_k, self.index.ntotal))
        id_list = list(self.metadata.keys())
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(id_list):
                continue
            vid = id_list[idx]
            results.append(
                type(
                    "Result",
                    (),
                    {
                        "id": vid,
                        "score": float(score),
                        "payload": self.metadata.get(vid, {}),
                    },
                )()
            )
        return results

    def delete(self, vector_id):
        vid = str(vector_id)
        if vid in self.metadata:
            del self.metadata[vid]
        logger.warning("FAISS does not support native deletion; metadata removed but index unchanged.")
        self._save()

    def update(self, vector_id, vector=None, payload=None):
        vid = str(vector_id)
        if payload:
            self.metadata[vid] = payload
        if vector:
            logger.warning("FAISS does not support native vector update; re-inserting may cause duplicates.")
        self._save()

    def get(self, vector_id):
        vid = str(vector_id)
        if vid in self.metadata:
            return type("Result", (), {"id": vid, "payload": self.metadata[vid]})()
        return None

    def list_cols(self):
        if os.path.exists(self.path):
            return [
                f.replace(".index", "")
                for f in os.listdir(self.path)
                if f.endswith(".index")
            ]
        return []

    def delete_col(self):
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        self.index = None
        self.metadata = {}

    def col_info(self):
        count = self.index.ntotal if self.index else 0
        return {"name": self.collection_name, "count": count}

    def list(self, filters=None, top_k=100):
        results = []
        for vid, payload in list(self.metadata.items())[:top_k]:
            results.append(type("Result", (), {"id": vid, "payload": payload})())
        return results

    def reset(self):
        self.delete_col()
        self.metadata = {}
