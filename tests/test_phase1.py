"""Phase 1 验证脚本 - 测试所有接口和配置类"""

import sys
sys.path.insert(0, ".")

# 1. 验证异常体系
from easy_memory.exceptions import (
    AgentMemoryError, AuthenticationError, RateLimitError, ValidationError,
    MemoryNotFoundError, NetworkError, ConfigurationError,
    VectorStoreError, EmbeddingError, LLMError, DatabaseError,
    create_exception_from_response,
)

# 测试基类异常
err = AgentMemoryError("test error", "TEST_001", suggestion="try again")
assert err.message == "test error"
assert err.error_code == "TEST_001"
assert err.suggestion == "try again"
print("[PASS] AgentMemoryError base class")

# 测试子类默认值
vs_err = VectorStoreError("connection failed")
assert vs_err.error_code == "VECTOR_001"
assert "vector store" in vs_err.suggestion.lower()
print("[PASS] VectorStoreError defaults")

# 测试工厂函数
auth_err = create_exception_from_response(401, "Unauthorized")
assert isinstance(auth_err, AuthenticationError)
assert auth_err.error_code == "HTTP_401"
print("[PASS] create_exception_from_response")

# 2. 验证抽象基类
from abc import ABC
from easy_memory.memory.base import MemoryBase
from easy_memory.llms.base import LLMBase
from easy_memory.embeddings.base import EmbeddingBase
from easy_memory.vector_stores.base import VectorStoreBase
from easy_memory.reranker.base import BaseReranker

# 确认它们是抽象类
assert issubclass(MemoryBase, ABC)
assert issubclass(LLMBase, ABC)
assert issubclass(EmbeddingBase, ABC)
assert issubclass(VectorStoreBase, ABC)
assert issubclass(BaseReranker, ABC)
print("[PASS] All base classes are ABCs")

# 确认不能直接实例化
try:
    MemoryBase()
    assert False, "Should have raised TypeError"
except TypeError:
    print("[PASS] MemoryBase cannot be instantiated directly")

try:
    VectorStoreBase()
    assert False, "Should have raised TypeError"
except TypeError:
    print("[PASS] VectorStoreBase cannot be instantiated directly")

# 3. 验证配置类
from easy_memory.configs.base import MemoryConfig, MemoryItem, AzureConfig
from easy_memory.configs.enums import MemoryType
from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.configs.llms.openai import OpenAIConfig
from easy_memory.configs.llms.anthropic import AnthropicConfig
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.vector_stores.configs import VectorStoreConfig
from easy_memory.configs.rerankers.base import BaseRerankerConfig
from easy_memory.configs.rerankers.config import RerankerConfig

# 测试MemoryItem
item = MemoryItem(id="1", memory="test memory")
assert item.id == "1"
assert item.memory == "test memory"
assert item.score is None
print("[PASS] MemoryItem creation")

# 测试MemoryConfig默认值
config = MemoryConfig()
assert config.llm.provider == "openai"
assert config.embedder.provider == "openai"
assert config.vector_store.provider == "qdrant"
assert config.version == "v1.1"
print("[PASS] MemoryConfig defaults")

# 测试枚举
assert MemoryType.SEMANTIC.value == "semantic_memory"
assert MemoryType.EPISODIC.value == "episodic_memory"
assert MemoryType.PROCEDURAL.value == "procedural_memory"
print("[PASS] MemoryType enum")

# 测试LLM配置
openai_config = OpenAIConfig(model="gpt-4o")
assert openai_config.model == "gpt-4o"
assert openai_config.temperature == 0.1
assert openai_config.max_tokens == 2000
print("[PASS] OpenAIConfig creation")

anthropic_config = AnthropicConfig(model="claude-3-5-sonnet")
assert anthropic_config.model == "claude-3-5-sonnet"
print("[PASS] AnthropicConfig creation")

# 测试BaseLlmConfig
base_llm = BaseLlmConfig(model="test-model", temperature=0.5)
assert base_llm.model == "test-model"
assert base_llm.temperature == 0.5
print("[PASS] BaseLlmConfig creation")

# 测试AzureConfig
azure = AzureConfig(api_key="test-key", azure_deployment="test-deploy")
assert azure.api_key == "test-key"
print("[PASS] AzureConfig creation")

# 测试RerankerConfig
reranker_config = RerankerConfig(provider="cohere")
assert reranker_config.provider == "cohere"
print("[PASS] RerankerConfig creation")

# 测试BaseRerankerConfig
br_config = BaseRerankerConfig(provider="test", model="test-model")
assert br_config.provider == "test"
print("[PASS] BaseRerankerConfig creation")

# 4. 验证Prompt模板
from easy_memory.configs.prompts import (
    MEMORY_ANSWER_PROMPT, FACT_RETRIEVAL_PROMPT, USER_MEMORY_EXTRACTION_PROMPT,
    AGENT_MEMORY_EXTRACTION_PROMPT, DEFAULT_UPDATE_MEMORY_PROMPT,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT, ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX, get_update_memory_messages, generate_additive_extraction_prompt,
)

assert len(MEMORY_ANSWER_PROMPT) > 0
assert len(FACT_RETRIEVAL_PROMPT) > 0
assert len(USER_MEMORY_EXTRACTION_PROMPT) > 0
assert len(AGENT_MEMORY_EXTRACTION_PROMPT) > 0
assert len(DEFAULT_UPDATE_MEMORY_PROMPT) > 0
assert len(PROCEDURAL_MEMORY_SYSTEM_PROMPT) > 0
assert len(ADDITIVE_EXTRACTION_PROMPT) > 0
print("[PASS] All prompt templates loaded")

# 测试get_update_memory_messages
msg = get_update_memory_messages(
    [{"id": "0", "text": "User likes pizza"}],
    '["Likes chicken pizza"]'
)
assert "memory" in msg
assert "User likes pizza" in msg
print("[PASS] get_update_memory_messages")

# 测试generate_additive_extraction_prompt
prompt = generate_additive_extraction_prompt(
    summary="Test summary",
    new_messages=[{"role": "user", "content": "Hello"}],
)
assert "Test summary" in prompt
assert "Hello" in prompt
print("[PASS] generate_additive_extraction_prompt")

# 测试带语言要求的prompt
prompt_cn = generate_additive_extraction_prompt(
    new_messages=[{"role": "user", "content": "你好"}],
    use_input_language=True,
)
assert "SAME LANGUAGE" in prompt_cn
print("[PASS] generate_additive_extraction_prompt with language requirement")

# 5. 验证LLMBase的推理模型检测
from easy_memory.llms.base import LLMBase

class TestLLM(LLMBase):
    def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
        return "test"

llm = TestLLM(config=BaseLlmConfig(model="gpt-4o"))
assert llm._is_reasoning_model("o1") == True
assert llm._is_reasoning_model("o3-mini") == True
assert llm._is_reasoning_model("gpt-4o") == False
assert llm._is_reasoning_model("openai/o1-preview") == True
print("[PASS] LLMBase reasoning model detection")

# 测试参数过滤
params = llm._get_supported_params(messages=[{"role": "user", "content": "test"}])
assert "temperature" in params  # 非推理模型包含temperature
print("[PASS] LLMBase parameter filtering")

# 推理模型测试
llm_reasoning = TestLLM(config=BaseLlmConfig(model="o1"))
params_r = llm_reasoning._get_supported_params(
    messages=[{"role": "user", "content": "test"}],
    temperature=0.5,  # 应该被过滤掉
)
assert "temperature" not in params_r
assert "messages" in params_r
print("[PASS] LLMBase reasoning model parameter filtering")

print("\n" + "=" * 50)
print("ALL PHASE 1 TESTS PASSED!")
print("=" * 50)