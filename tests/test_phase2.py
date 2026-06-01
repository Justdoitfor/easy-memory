"""测试工厂模式和 Provider 实现。

运行方式：
    python verify_phase2.py

前提条件：
    - 安装依赖: pip install openai qdrant-client
    - 如需测试真实 API: 设置 OPENAI_API_KEY 环境变量
"""
import sys

sys.path.insert(0, ".")

from easy_memory.utils.factory import LlmFactory, EmbedderFactory, VectorStoreFactory, load_class


def test_load_class():
    """测试动态类加载。"""
    print("=== 测试 load_class ===")

    # 测试加载 MockEmbeddings
    cls = load_class("easy_memory.embeddings.mock.MockEmbeddings")
    instance = cls()
    embedding = instance.embed("test text")
    print(f"MockEmbeddings 加载成功，输出: {embedding}")
    assert len(embedding) == 10, "MockEmbeddings 应返回 10 维向量"

    print("load_class 测试通过!\n")


def test_llm_factory():
    """测试 LLM 工厂。"""
    print("=== 测试 LlmFactory ===")

    # 列出支持的 provider
    providers = LlmFactory.get_supported_providers()
    print(f"支持的 LLM providers: {providers}")
    assert "openai" in providers

    # 测试创建 OpenAI LLM（需要 API key）
    import os
    if os.getenv("OPENAI_API_KEY"):
        llm = LlmFactory.create("openai", {"model": "gpt-4o-mini"})
        print(f"OpenAI LLM 创建成功: {type(llm).__name__}")
        print(f"模型: {llm.config.model}")
    else:
        print("跳过 OpenAI LLM 测试（未设置 OPENAI_API_KEY）")

    # 测试不支持的 provider
    try:
        LlmFactory.create("nonexistent")
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"预期的错误: {e}")

    print("LlmFactory 测试通过!\n")


def test_embedder_factory():
    """测试 Embedder 工厂。"""
    print("=== 测试 EmbedderFactory ===")

    # 测试 MockEmbeddings
    mock = EmbedderFactory.create("openai", {"model": "mock"}, None)
    # 注意：这需要 openai API key，这里仅测试工厂能正确创建

    print("EmbedderFactory 测试通过!\n")


def test_vector_store_factory():
    """测试 VectorStore 工厂。"""
    print("=== 测试 VectorStoreFactory ===")

    # 测试 Qdrant（本地模式）
    try:
        store = VectorStoreFactory.create("qdrant", {
            "collection_name": "test_collection",
            "embedding_model_dims": 10,
            "path": "/tmp/test_qdrant",
        })
        print(f"Qdrant 创建成功: {type(store).__name__}")
        print(f"集合名称: {store.collection_name}")

        # 测试插入和搜索
        store.insert(
            vectors=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
            payloads=[{"text": "hello world", "user_id": "test"}],
            ids=["test_1"],
        )
        print("插入测试数据成功")

        results = store.search(
            query="test",
            vectors=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            top_k=1,
        )
        print(f"搜索返回 {len(results)} 条结果")
        assert len(results) == 1

        # 清理
        store.delete_col()
        print("清理完成")
    except ImportError:
        print("跳过 Qdrant 测试（未安装 qdrant-client）")

    # 测试不支持的 provider
    try:
        VectorStoreFactory.create("nonexistent", {})
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"预期的错误: {e}")

    print("VectorStoreFactory 测试通过!\n")


def test_register_provider():
    """测试运行时注册 provider。"""
    print("=== 测试 register_provider ===")

    # 注册自定义 LLM provider
    LlmFactory.register_provider(
        name="custom_llm",
        class_path="easy_memory.llms.openai.OpenAILLM",
        config_class=None,
    )
    providers = LlmFactory.get_supported_providers()
    assert "custom_llm" in providers
    print(f"注册自定义 provider 'custom_llm' 成功")
    print(f"当前支持的 providers: {providers}")

    print("register_provider 测试通过!\n")


if __name__ == "__main__":
    print("Phase 2 验证开始...\n")
    test_load_class()
    test_llm_factory()
    test_embedder_factory()
    test_vector_store_factory()
    test_register_provider()
    print("所有测试完成!")
