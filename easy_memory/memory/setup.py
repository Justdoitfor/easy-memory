import json
import logging
import os
import uuid
from hashlib import sha256

VECTOR_ID = str(uuid.uuid4())
home_dir = os.path.expanduser("~")
easy_memory_dir = os.environ.get("EASY_MEMORY_DIR") or os.path.join(home_dir, ".easy_memory")
os.makedirs(easy_memory_dir, exist_ok=True)

_logger = logging.getLogger(__name__)


def _config_path():
    return os.path.join(easy_memory_dir, "config.json")


def _load_config():
    path = _config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        _logger.debug("Failed to load config %s: %s", path, e)
        return {}


def _write_config(config):
    path = _config_path()
    try:
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        _logger.debug("Failed to write config %s: %s", path, e)


def setup_config():
    config = _load_config()
    if config.get("user_id"):
        return
    config["user_id"] = str(uuid.uuid4())
    _write_config(config)


def get_user_id():
    config = _load_config()
    if not config:
        return "anonymous_user"
    return config.get("user_id")


def read_anon_ids():
    config = _load_config()
    telemetry = config.get("telemetry") if isinstance(config.get("telemetry"), dict) else {}
    aliased_pairs = telemetry.get("aliased_pairs")
    return {
        "oss": config.get("user_id"),
        "cli": telemetry.get("anonymous_id"),
        "aliased_pairs": aliased_pairs if isinstance(aliased_pairs, list) else [],
    }


def _alias_pair_marker(anon_id, email):
    return sha256(f"{anon_id}\0{email}".encode("utf-8")).hexdigest()


def is_aliased(anon_id, email):
    if not anon_id or not email:
        return False
    config = _load_config()
    telemetry = config.get("telemetry") if isinstance(config.get("telemetry"), dict) else {}
    aliased_pairs = telemetry.get("aliased_pairs")
    if not isinstance(aliased_pairs, list):
        return False
    return _alias_pair_marker(anon_id, email) in aliased_pairs


def mark_aliased(anon_id, email):
    if not anon_id or not email:
        return
    config = _load_config()
    telemetry = config.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}
    aliased_pairs = telemetry.get("aliased_pairs")
    if not isinstance(aliased_pairs, list):
        aliased_pairs = []
    marker = _alias_pair_marker(anon_id, email)
    if marker not in aliased_pairs:
        aliased_pairs.append(marker)
    telemetry["aliased_pairs"] = aliased_pairs
    config["telemetry"] = telemetry
    _write_config(config)


def get_or_create_user_id(vector_store=None):
    user_id = get_user_id()
    if vector_store is None:
        return user_id
    try:
        existing = vector_store.get(vector_id=user_id)
        if existing and hasattr(existing, "payload") and existing.payload and "user_id" in existing.payload:
            stored_id = existing.payload["user_id"]
            if stored_id is not None:
                return stored_id
    except Exception:
        pass
    try:
        dims = getattr(vector_store, "embedding_model_dims", 1536)
        vector_store.insert(vectors=[[0.1] * dims], payloads=[{"user_id": user_id, "type": "user_identity"}],
                            ids=[user_id])
    except Exception:
        pass
    return user_id
