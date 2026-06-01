"""
端到端测试脚本 - 验证 Phase 3 记忆引擎

使用前请确保已配置好 LLM 和 Embedding 模型（见 Phase 2 配置文件）。
"""

import json
import sys

# 确保项目在 Python 路径中
sys.path.insert(0, ".")


def test_memory_basic():
    """测试基本的 add/search/get/update/delete 流程"""
    from easy_memory.memory.main import Memory
    from easy_memory.configs.base import MemoryConfig

    print("=" * 60)
    print("测试1: 基本 CRUD 操作")
    print("=" * 60)

    # 初始化
    config = MemoryConfig()
    m = Memory(config)

    try:
        # Add - 添加记忆
        print("\n[1] 添加记忆...")
        result = m.add(
            "我喜欢吃火锅，特别是重庆火锅",
            user_id="test_user_001"
        )
        print(f"  添加结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

        # Search - 搜索记忆
        print("\n[2] 搜索记忆...")
        search_result = m.search(
            "用户喜欢吃什么？",
            filters={"user_id": "test_user_001"},
            top_k=5
        )
        print(f"  搜索结果: {json.dumps(search_result, ensure_ascii=False, indent=2)}")

        # Get All - 获取所有记忆
        print("\n[3] 获取所有记忆...")
        all_memories = m.get_all(filters={"user_id": "test_user_001"})
        print(f"  总记忆数: {len(all_memories['results'])}")

        if all_memories["results"]:
            memory_id = all_memories["results"][0]["id"]

            # Get - 获取单条记忆
            print(f"\n[4] 获取单条记忆 (id={memory_id})...")
            single = m.get(memory_id)
            print(f"  记忆内容: {single}")

            # History - 查看历史
            print(f"\n[5] 查看记忆历史...")
            hist = m.history(memory_id)
            print(f"  历史记录数: {len(hist)}")
            for h in hist:
                print(f"    事件: {h['event']}, 内容: {h['new_memory'][:50]}...")

            # Update - 更新记忆
            print(f"\n[6] 更新记忆...")
            update_result = m.update(memory_id, "我喜欢吃川菜，尤其是麻辣火锅")
            print(f"  更新结果: {update_result}")

            # 再次查看历史
            print(f"\n[7] 更新后的历史...")
            hist = m.history(memory_id)
            for h in hist:
                print(
                    f"    事件: {h['event']}, 旧值: {str(h['old_memory'])[:30]}..., 新值: {str(h['new_memory'])[:30]}...")

            # Delete - 删除记忆
            print(f"\n[8] 删除记忆...")
            delete_result = m.delete(memory_id)
            print(f"  删除结果: {delete_result}")

            # 确认删除
            print(f"\n[9] 确认删除...")
            deleted = m.get(memory_id)
            print(f"  获取已删除记忆: {deleted}")

    finally:
        m.close()
        print("\n✓ 基本 CRUD 测试完成")


def test_memory_pipeline():
    """测试 V3 流水线的 infer 模式"""
    from easy_memory.memory.main import Memory
    from easy_memory.configs.base import MemoryConfig

    print("\n" + "=" * 60)
    print("测试2: V3 流水线 (infer 模式)")
    print("=" * 60)

    config = MemoryConfig()
    m = Memory(config)

    try:
        # 添加对话消息
        messages = [
            {"role": "user", "content": "我叫张三，在北京工作"},
            {"role": "assistant", "content": "你好张三！北京是个好城市。"},
            {"role": "user", "content": "我是一名软件工程师，喜欢用Python"},
        ]

        print("\n[1] 通过对话添加记忆 (infer=True)...")
        result = m.add(messages, user_id="test_user_002")
        print(f"  提取到 {len(result['results'])} 条记忆:")
        for mem in result["results"]:
            print(f"    - {mem['memory']}")

        # 搜索验证
        print("\n[2] 搜索: 用户的职业是什么？")
        search_result = m.search(
            "用户的职业",
            filters={"user_id": "test_user_002"},
            top_k=3
        )
        for mem in search_result["results"]:
            print(f"    [{mem['score']:.3f}] {mem['memory']}")

        # 追加对话 - 测试增量提取
        print("\n[3] 追加对话，测试增量提取...")
        new_messages = [
            {"role": "user", "content": "我最近搬到了上海"},
        ]
        result2 = m.add(new_messages, user_id="test_user_002")
        print(f"  新增 {len(result2['results'])} 条记忆:")
        for mem in result2["results"]:
            print(f"    - {mem['memory']}")

        # 验证最终状态
        print("\n[4] 最终记忆列表:")
        all_memories = m.get_all(filters={"user_id": "test_user_002"})
        for mem in all_memories["results"]:
            print(f"    - {mem['memory']}")

    finally:
        m.close()
        print("\n✓ V3 流水线测试完成")


def test_multi_user_isolation():
    """测试多用户记忆隔离"""
    from easy_memory.memory.main import Memory
    from easy_memory.configs.base import MemoryConfig

    print("\n" + "=" * 60)
    print("测试3: 多用户隔离")
    print("=" * 60)

    config = MemoryConfig()
    m = Memory(config)

    try:
        # 用户A添加记忆
        m.add("我喜欢猫", user_id="user_a")
        m.add("我住在上海", user_id="user_a")

        # 用户B添加记忆
        m.add("我喜欢狗", user_id="user_b")
        m.add("我住在北京", user_id="user_b")

        # 搜索 - 应该只能看到自己的记忆
        print("\n[1] 用户A搜索 '喜欢什么动物':")
        result_a = m.search("喜欢什么动物", filters={"user_id": "user_a"})
        for mem in result_a["results"]:
            print(f"    [{mem['score']:.3f}] {mem['memory']}")

        print("\n[2] 用户B搜索 '喜欢什么动物':")
        result_b = m.search("喜欢什么动物", filters={"user_id": "user_b"})
        for mem in result_b["results"]:
            print(f"    [{mem['score']:.3f}] {mem['memory']}")

        # 清理
        m.delete_all(user_id="user_a")
        m.delete_all(user_id="user_b")

    finally:
        m.close()
        print("\n✓ 多用户隔离测试完成")


if __name__ == "__main__":
    test_memory_basic()
    test_memory_pipeline()
    test_multi_user_isolation()
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
