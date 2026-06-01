# `ABC`（Abstract Base Class）是Python内置的抽象基类机制。
# 标记了 `@abstractmethod` 的方法 必须 被子类实现，否则实例化时会抛出 `TypeError`

from abc import ABC, abstractmethod


class MemoryBase(ABC):
    """定义记忆存储的核心操作接口。所有记忆存储实现（如基于SQL的、基于文件的）都必须实现这5个方法"""

    @abstractmethod
    def get(self, memory_id):
        pass

    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def update(self, memory_id, data):
        pass

    @abstractmethod
    def delete(self, memory_id):
        pass

    @abstractmethod
    def history(self, memory_id):
        pass
