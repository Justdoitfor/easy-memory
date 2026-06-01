from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self,
               query: str,
               document: List[Dict[str, Any]],
               top_k: int = None) -> List[Dict[str, Any]]:
        pass

