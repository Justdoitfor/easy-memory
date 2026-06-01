"""
定义了向量数据库的完整操作接口，包括：
集合管理（`create_col`、`list_cols`、`delete_col`、`col_info`）、
数据操作（`insert`、`update`、`delete`、`get`）、
搜索（`search`、`keyword_search`、`search_batch`）
"""
from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    @abstractmethod
    def create_col(self, vector_size, distance):
        pass

    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None):
        pass

    @abstractmethod
    def search(self, query, vectors, top_k=5, filters=None):
        pass

    @abstractmethod
    def delete(self, vector_id):
        pass

    @abstractmethod
    def update(self, vector_id, vector=None, payloads=None):
        pass

    @abstractmethod
    def get(self, vector_id):
        pass

    @abstractmethod
    def list_cols(self):
        pass

    @abstractmethod
    def delete_col(self):
        pass

    @abstractmethod
    def col_info(self):
        pass

    @abstractmethod
    def list(self, filters=None, top_k=None):
        pass

    @abstractmethod
    def reset(self):
        pass

    def keyword_search(self, query: str, top_k: int = 5, filters: dict = None):
        return None

    def search_batch(self, queries: list, vectors_list: list, top_k: int = 1, filters: dict = None):
        return [self.search(q, v, top_k=top_k, filters=filters) for q, v in zip(queries, vectors_list)]
