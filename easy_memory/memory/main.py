import hashlib
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import ValidationError

from easy_memory.configs.base import MemoryConfig, MemoryItem
from easy_memory.configs.enums import MemoryType
from easy_memory.configs.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    generate_additive_extraction_prompt,
)
from easy_memory.exceptions import ValidationError as Mem0ValidationError
from easy_memory.memory.base import MemoryBase
from easy_memory.memory.setup import easy_memory_dir, setup_config
from easy_memory.memory.storage import SQLiteManager
from easy_memory.memory.utils import (
    extract_json,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from easy_memory.utils.factory import (
    EmbedderFactory,
    LlmFactory,
    RerankerFactory,
    VectorStoreFactory,
)
from easy_memory.utils.entity_extraction import extract_entities, extract_entities_batch
from easy_memory.utils.lemmatization import lemmatize_for_bm25
from easy_memory.memory.telemetry import MEMO_TELEMETRY, capture_event
from easy_memory.utils.scoring import ENTITY_BOOST_WEIGHT, get_bm25_params, normalize_bm25, score_and_rank

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

logger = logging.getLogger(__name__)

_RUNTIME_FIELDS = frozenset({"http_auth", "auth", "connection_class", "ssl_context"})
_SENSITIVE_FIELDS_EXACT = frozenset({
    "api_key", "secret_key", "private_key", "access_key", "password",
    "credentials", "credential", "secret", "token", "access_token",
    "refresh_token", "auth_token", "session_token", "client_secret",
    "auth_client_secret", "azure_client_secret", "service_account_json", "aws_session_token",
})
_SENSITIVE_SUFFIXES = ("_password", "_secret", "_token", "_credential", "_credentials")
ENTITY_PARAMS = frozenset({"user_id", "agent_id", "run_id"})


def _reject_top_level_entity_params(kwargs, method_name):
    invalid_keys = ENTITY_PARAMS & set(kwargs.keys())
    if invalid_keys:
        raise ValueError(
            f"Top-level entity parameters {invalid_keys} are not supported in {method_name}(). "
            f"Use filters={{'user_id': '...'}} instead."
        )


def _validate_and_trim_entity_id(value, name):
    if value is None:
        return None
    trimmed = value.strip()
    if trimmed == "":
        raise ValueError(f"Invalid {name}: cannot be empty or whitespace-only.")
    if any(c.isspace() for c in trimmed):
        raise ValueError(f"Invalid {name}: cannot contain whitespace.")
    return trimmed


def _validate_search_params(threshold=None, top_k=None):
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a valid number")
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a valid integer")
        if top_k < 0:
            raise ValueError(f"Invalid top_k: {top_k}. Must be a non-negative integer.")


def _is_sensitive_field(field_name):
    name = field_name.lower().strip()
    if name in _RUNTIME_FIELDS:
        return False
    if name in _SENSITIVE_FIELDS_EXACT:
        return True
    return any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def _safe_deepcopy_config(config):
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"Deepcopy failed, using dict-based cloning: {e}")
        config_class = type(config)
        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump()
            except Exception:
                clone_dict = dict(config.__dict__)
        else:
            clone_dict = dict(config.__dict__)
        for field_name in list(clone_dict.keys()):
            if field_name in _RUNTIME_FIELDS and hasattr(config, field_name):
                clone_dict[field_name] = getattr(config, field_name)
            elif _is_sensitive_field(field_name):
                clone_dict[field_name] = None
        try:
            return config_class(**clone_dict)
        except Exception:
            return type("Config", (), clone_dict)()


def _normalize_iso_timestamp_to_utc(timestamp):
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


def _build_filters_and_metadata(*, user_id=None, agent_id=None, run_id=None,
                                actor_id=None, input_metadata=None, input_filters=None):
    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    session_ids_provided = []
    user_id = _validate_and_trim_entity_id(user_id, "user_id")
    agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
    run_id = _validate_and_trim_entity_id(run_id, "run_id")

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")
    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")
    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation.",
        )

    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


def _build_session_scope(filters):
    parts = []
    for key in sorted(["user_id", "agent_id", "run_id"]):
        val = filters.get(key)
        if val:
            parts.append(f"{key}={val}")
    return "&".join(parts)


setup_config()


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions

        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(config.reranker.provider, config.reranker.config)
        self._entity_store = None
        capture_event("mem0.init", self, {"sync_type": "sync"})

    @classmethod
    def from_config(cls, config_dict):
        try:
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    def _should_use_agent_memory_extraction(self, messages, metadata):
        has_agent_id = metadata.get("agent_id") is not None
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)
        return has_agent_id and has_assistant_messages

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            entity_collection = f"{self.collection_name}_entities"
            if hasattr(entity_config, 'collection_name'):
                entity_config.collection_name = entity_collection
            elif isinstance(entity_config, dict):
                entity_config['collection_name'] = entity_collection
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                if hasattr(entity_config, "client"):
                    entity_config.client = self.vector_store.client
                elif isinstance(entity_config, dict):
                    entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    def _upsert_entity(self, entity_text, entity_type, memory_id, filters):
        """Upsert an entity into the entity store, linking it to a memory."""
        try:
            entity_embedding = self.embedding_model.embed(entity_text, "add")
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
            existing = self.entity_store.search(
                query=entity_text, vectors=entity_embedding, top_k=1, filters=search_filters,
            )
            if existing and existing[0].score >= 0.95:
                match = existing[0]
                payload = match.payload or {}
                linked_ids = payload.get("linked_memory_ids", [])
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    self.entity_store.update(vector_id=match.id, vector=None, payload=payload)
            else:
                entity_id = str(uuid.uuid4())
                entity_payload = {
                    "data": entity_text, "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                self.entity_store.insert(vectors=[entity_embedding], ids=[entity_id], payloads=[entity_payload])
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}': {e}")

    def _remove_memory_from_entity_store(self, memory_id, filters):
        """Strip memory_id from entity records."""
        if self._entity_store is None:
            return
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = self.entity_store.list(filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        try:
                            self.entity_store.delete(vector_id=row.id)
                        except Exception as e:
                            logger.debug(f"Entity delete failed for id={row.id}: {e}")
                    else:
                        entity_text = payload.get("data")
                        if not isinstance(entity_text, str) or not entity_text:
                            continue
                        try:
                            vec = self.embedding_model.embed(entity_text, "update")
                        except Exception:
                            continue
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            self.entity_store.update(vector_id=row.id, vector=vec, payload=new_payload)
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id}: {e}")
                except Exception as e:
                    logger.debug(f"Entity cleanup error: {e}")
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id}: {e}")

    def _link_entities_for_memory(self, memory_id, text, filters):
        """Extract entities from text and link them to memory_id."""
        try:
            entities = extract_entities(text)
            if not entities:
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = entity_text.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    self._upsert_entity(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}': {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id}: {e}")

    def _compute_entity_boosts(self, query_entities, filters):
        """Compute per-memory entity boosts from entity store search."""
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = entity_text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))
        if not deduped:
            return {}
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}
        try:
            for _, entity_text in deduped:
                entity_embedding = self.embedding_model.embed(entity_text, "search")
                matches = self.entity_store.search(
                    query=entity_text, vectors=entity_embedding, top_k=500, filters=search_filters,
                )
                for match in matches:
                    similarity = match.score if hasattr(match, 'score') else 0.0
                    if similarity < 0.5:
                        continue
                    payload = match.payload if hasattr(match, 'payload') else {}
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    if not isinstance(linked_memory_ids, list):
                        continue
                    num_linked = max(len(linked_memory_ids), 1)
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight
                    for memory_id in linked_memory_ids:
                        if memory_id:
                            memory_key = str(memory_id)
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)
        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")
        return memory_boosts

    def add(self, messages, *, user_id=None, agent_id=None, run_id=None,
            metadata=None, infer=True, memory_type=None, prompt=None):
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata,
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories.",
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries.",
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            return self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        vector_store_result = self._add_to_vector_store(messages, processed_metadata, effective_filters, infer,
                                                        prompt=prompt)
        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer, prompt=None):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if not isinstance(message_dict, dict) or message_dict.get("role") is None or message_dict.get(
                        "content") is None:
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue
                if message_dict["role"] == "system":
                    continue
                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]
                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name
                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                mem_id = self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)
                returned_memories.append({
                    "id": mem_id, "memory": msg_content, "event": "ADD",
                    "actor_id": actor_name if actor_name else None, "role": message_dict["role"],
                })
            return returned_memories

        # === V3 PHASED BATCH PIPELINE ===

        # Phase 0: Context gathering
        session_scope = _build_session_scope(filters)
        last_messages = self.db.get_last_messages(session_scope, limit=10)
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        query_embedding = self.embedding_model.embed(parsed_messages, "search")
        existing_results = self.vector_store.search(
            query=parsed_messages, vectors=query_embedding, top_k=10, filters=search_filters,
        )

        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction
        is_agent_scoped = bool(filters.get("agent_id")) and not filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX

        custom_instr = prompt or self.custom_instructions
        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories, new_messages=parsed_messages,
            last_k_messages=last_messages, custom_instructions=custom_instr,
        )

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            extracted_memories = []

        if not extracted_memories:
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 3: Batch embed
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        except Exception:
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = self.embedding_model.embed(text, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed memory text: {e}")

        # Phase 4+5: CPU processing + hash dedup
        existing_hashes = set()
        for mem in existing_results:
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            if h:
                existing_hashes.add(h)

        records = []
        seen_hashes = set()
        for mem in extracted_memories:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue
            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match): {text[:50]}")
                continue
            seen_hashes.add(mem_hash)

            text_lemmatized = lemmatize_for_bm25(text)
            memory_id = str(uuid.uuid4())
            mem_metadata = deepcopy(metadata)
            mem_metadata["data"] = text
            mem_metadata["text_lemmatized"] = text_lemmatized
            mem_metadata["hash"] = mem_hash
            if "created_at" not in mem_metadata:
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            if mem.get("attributed_to"):
                mem_metadata["attributed_to"] = mem["attributed_to"]
            records.append((memory_id, text, embed_map[text], mem_metadata))

        if not records:
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]

        try:
            self.vector_store.insert(vectors=all_vectors, ids=all_ids, payloads=all_payloads)
        except Exception:
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                try:
                    self.vector_store.insert(vectors=[vec], ids=[mid], payloads=[pay])
                except Exception as e:
                    logger.error(f"Failed to insert memory {mid}: {e}")

        history_records = [
            {"memory_id": r[0], "old_memory": None, "new_memory": r[1], "event": "ADD",
             "created_at": r[3].get("created_at"), "is_deleted": 0}
            for r in records
        ]
        try:
            self.db.batch_add_history(history_records)
        except Exception:
            for hr in history_records:
                try:
                    self.db.add_history(hr["memory_id"], None, hr["new_memory"], "ADD", created_at=hr.get("created_at"))
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']}: {e}")

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in records]
            all_entities = extract_entities_batch(all_texts)
            global_entities = {}
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text in entities:
                    key = entity_text.strip().lower()
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}]
            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]
                try:
                    entity_embeddings = self.embedding_model.embed_batch(entity_texts, "add")
                except Exception:
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(self.embedding_model.embed(t, "add"))
                        except Exception:
                            entity_embeddings.append(None)
                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    existing_matches = self.entity_store.search_batch(
                        queries=valid_texts, vectors_list=valid_vectors, top_k=1, filters=search_filters,
                    )
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    for j, key in enumerate(valid_keys):
                        entity_type, entity_text, memory_ids = global_entities[key]
                        matches = existing_matches[j] if j < len(existing_matches) else []
                        if matches and matches[0].score >= 0.95:
                            match = matches[0]
                            payload = match.payload or {}
                            linked = set(payload.get("linked_memory_ids", []))
                            linked |= memory_ids
                            payload["linked_memory_ids"] = sorted(linked)
                            try:
                                self.entity_store.update(vector_id=match.id, vector=None, payload=payload)
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}': {e}")
                        else:
                            to_insert_vectors.append(valid_vectors[j])
                            to_insert_ids.append(str(uuid.uuid4()))
                            to_insert_payloads.append({
                                "data": entity_text, "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids), **search_filters,
                            })
                    if to_insert_vectors:
                        try:
                            self.entity_store.insert(vectors=to_insert_vectors, ids=to_insert_ids,
                                                     payloads=to_insert_payloads)
                        except Exception as e:
                            logger.warning(f"Batch entity insert failed: {e}")
        except Exception as e:
            logger.warning(f"Batch entity linking failed: {e}")

        # Phase 8: Save messages
        self.db.save_messages(messages, session_scope)

        returned_memories = [{"id": r[0], "memory": r[1], "event": "ADD"} for r in records]
        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.add", self,
                      {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        return returned_memories

    def get(self, memory_id):
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None
        promoted_payload_keys = ["user_id", "agent_id", "run_id", "actor_id", "role"]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to",
                                  *promoted_payload_keys}
        result_item = MemoryItem(
            id=memory.id, memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"), created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()
        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata
        return result_item

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        capture_event("mem0.get_all", self, {"limit": top_k, "sync_type": "sync"})
        _reject_top_level_entity_params(kwargs, "get_all")
        _validate_search_params(top_k=top_k)
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(effective_filters["user_id"], "user_id")
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(effective_filters["agent_id"], "agent_id")
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(effective_filters["run_id"], "run_id")
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("filters must contain at least one of: user_id, agent_id, run_id.")
        return {"results": self._get_all_from_vector_store(effective_filters, top_k)}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, top_k=limit)
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = ["user_id", "agent_id", "run_id", "actor_id", "role"]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to",
                                  *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id, memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"), created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})
            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]
            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata
            formatted_memories.append(memory_item_dict)
        return formatted_memories

    def _has_advanced_operators(self, filters):
        if not isinstance(filters, dict):
            return False
        for key, value in filters.items():
            if key in ["AND", "OR", "NOT"]:
                return True
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            if value == "*":
                return True
        return False

    def _process_metadata_filters(self, metadata_filters):
        processed_filters = {}

        def process_condition(key, condition):
            if not isinstance(condition, dict):
                if condition == "*":
                    return {key: "*"}
                return {key: condition}
            result = {}
            for operator, value in condition.items():
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains",
                }
                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        def merge_filters(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    target[key].update(value)
                else:
                    target[key] = value

        for key, value in metadata_filters.items():
            if key == "AND":
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            elif key == "OR":
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                merge_filters(processed_filters, process_condition(key, value))
        return processed_filters

    def search(self, query, *, top_k=20, filters=None, threshold=0.1, rerank=False, **kwargs):
        capture_event("mem0.search", self,
                      {"limit": top_k, "version": self.api_version, "sync_type": "sync", "threshold": threshold})
        _reject_top_level_entity_params(kwargs, "search")
        _validate_search_params(threshold=threshold, top_k=top_k)
        effective_filters = filters.copy() if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(effective_filters["user_id"], "user_id")
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(effective_filters["agent_id"], "agent_id")
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(effective_filters["run_id"], "run_id")
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("filters must contain at least one of: user_id, agent_id, run_id.")
        if self._has_advanced_operators(effective_filters):
            processed_filters = self._process_metadata_filters(effective_filters)
            for logical_key in ("AND", "OR", "NOT"):
                effective_filters.pop(logical_key, None)
            for fk in list(effective_filters.keys()):
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(
                        effective_filters.get(fk), dict):
                    effective_filters.pop(fk, None)
            effective_filters.update(processed_filters)
        original_memories = self._search_vector_store(query, effective_filters, top_k, threshold)
        if rerank and self.reranker and original_memories:
            try:
                original_memories = self.reranker.rerank(query, original_memories, top_k)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
        return {"results": original_memories}

    def _search_vector_store(self, query, filters, limit, threshold=0.1):
        if threshold is None:
            threshold = 0.1
        query_lemmatized = lemmatize_for_bm25(query)
        query_entities = extract_entities(query)
        embeddings = self.embedding_model.embed(query, "search")
        internal_limit = max(limit * 4, 60)
        semantic_results = self.vector_store.search(
            query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )
        keyword_results = self.vector_store.keyword_search(
            query=query_lemmatized, top_k=internal_limit, filters=filters
        )
        bm25_scores = {}
        if keyword_results is not None:
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)
        entity_boosts = {}
        if query_entities:
            entity_boosts = self._compute_entity_boosts(query_entities, filters)
        candidates = []
        for mem in semantic_results:
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id, "score": mem.score,
                "payload": mem.payload if hasattr(mem, 'payload') else {},
            })
        scored_results = score_and_rank(
            semantic_results=candidates, bm25_scores=bm25_scores,
            entity_boosts=entity_boosts, threshold=threshold, top_k=limit,
        )
        promoted_payload_keys = ["user_id", "agent_id", "run_id", "actor_id", "role"]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to",
                                  *promoted_payload_keys}
        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}
            if not payload.get("data"):
                continue
            memory_item_dict = MemoryItem(
                id=scored["id"], memory=payload.get("data", ""),
                hash=payload.get("hash"), created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"), score=scored["score"],
            ).model_dump()
            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]
            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)
            original_memories.append(memory_item_dict)
        return original_memories

    def update(self, memory_id, data, metadata=None):
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})
        existing_embeddings = {data: self.embedding_model.embed(data, "update")}
        self._update_memory(memory_id, data, existing_embeddings, metadata)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})
        existing_memory = self.vector_store.get(vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")
        self._delete_memory(memory_id, existing_memory)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id=None, agent_id=None, run_id=None):
        capture_event("mem0.delete_all", self, {"sync_type": "sync"})
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id
        if not filters:
            raise ValueError("At least one filter is required. Use reset() to delete all.")
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)
        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        return self.db.get_history(memory_id)

    def _create_memory(self, data, existing_embeddings, metadata=None):
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        self.vector_store.insert(vectors=[embeddings], ids=[memory_id], payloads=[new_metadata])
        self.db.add_history(
            memory_id, None, data, "ADD",
            created_at=new_metadata.get("created_at"), updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"), role=new_metadata.get("role"),
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]
        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory: {e}")
            raise
        if metadata is None:
            raise ValueError("Metadata cannot be None for procedural memory.")
        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            raise ValueError(f"Error getting memory with ID {memory_id}.")
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found.")
        prev_value = existing_memory.payload.get("data")
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")
        self.vector_store.update(vector_id=memory_id, vector=embeddings, payload=new_metadata)
        self.db.add_history(
            memory_id, prev_value, data, "UPDATE",
            created_at=new_metadata["created_at"], updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"), role=new_metadata.get("role"),
        )
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        self._remove_memory_from_entity_store(memory_id, session_filters)
        self._link_entities_for_memory(memory_id, data, session_filters)
        return memory_id

    def _delete_memory(self, memory_id, existing_memory=None):
        if existing_memory is None:
            existing_memory = self.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found.")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id, prev_value, None, "DELETE",
            created_at=created_at, updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"), is_deleted=1,
        )
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}
        self._remove_memory_from_entity_store(memory_id, session_filters)
        return memory_id

    def reset(self):
        capture_event("mem0.reset", self, {"sync_type": "sync"})
        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.close()
        self.db = SQLiteManager(self.config.history_db_path)
        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )

    def close(self):
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(config.reranker.provider, config.reranker.config)
        self._entity_store = None
        self._sync = Memory(config)

    async def add(self, messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, infer=True,
                  memory_type=None, prompt=None):
        import asyncio
        return await asyncio.to_thread(self._sync.add, messages, user_id=user_id, agent_id=agent_id, run_id=run_id,
                                       metadata=metadata, infer=infer, memory_type=memory_type, prompt=prompt)

    async def get(self, memory_id):
        import asyncio
        return await asyncio.to_thread(self._sync.get, memory_id)

    async def get_all(self, *, filters=None, top_k=20, **kwargs):
        import asyncio
        return await asyncio.to_thread(self._sync.get_all, filters=filters, top_k=top_k, **kwargs)

    async def search(self, query, *, top_k=20, filters=None, threshold=0.1, rerank=False, **kwargs):
        import asyncio
        return await asyncio.to_thread(self._sync.search, query, top_k=top_k, filters=filters, threshold=threshold,
                                       rerank=rerank, **kwargs)

    async def update(self, memory_id, data, metadata=None):
        import asyncio
        return await asyncio.to_thread(self._sync.update, memory_id, data, metadata)

    async def delete(self, memory_id):
        import asyncio
        return await asyncio.to_thread(self._sync.delete, memory_id)

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        import asyncio
        return await asyncio.to_thread(self._sync.delete_all, user_id=user_id, agent_id=agent_id, run_id=run_id)

    async def history(self, memory_id):
        import asyncio
        return await asyncio.to_thread(self._sync.history, memory_id)

    async def reset(self):
        import asyncio
        return await asyncio.to_thread(self._sync.reset)

    async def close(self):
        import asyncio
        return await asyncio.to_thread(self._sync.close)
