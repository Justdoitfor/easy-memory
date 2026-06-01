"""
- 懒加载 + 双重检查锁：模型只在首次使用时加载，通过 _lock 保证线程安全
- 两个模型实例：get_nlp_full() 完整模型（含NER和Parser），get_nlp_lemma() 轻量模型（只做词形还原）
- 自动下载：如果 spaCy 模型未安装，自动尝试下载
- 失败缓存：加载失败后不再重试（_load_failed_* 标志），避免反复报错
"""
import logging
import threading

logger = logging.getLogger(__name__)

_nlp_full = None
_nlp_lemma = None
_load_failed_full = False
_load_failed_lemma = False
_lock = threading.Lock()


def _ensure_model_available():
    try:
        import spacy
    except ImportError:
        raise ImportError("spaCy is not installed. Install it with: pip install easy-memory[nlp]")
    if not spacy.util.is_package("en_core_web_sm"):
        logger.info("Downloading spaCy model en_core_web_sm...")
        try:
            from spacy.cli import download
            download("en_core_web_sm")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download spaCy model: {e}. Please install manually: python -m spacy download en_core_web_sm") from e


def get_nlp_full():
    global _nlp_full, _load_failed_full
    if _load_failed_full:
        return None
    if _nlp_full is not None:
        return _nlp_full
    with _lock:
        if _nlp_full is not None:
            return _nlp_full
        if _load_failed_full:
            return None
        try:
            _ensure_model_available()
            import spacy
            _nlp_full = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"Failed to load spaCy full model: {e}")
            _load_failed_full = True
            return None
    return _nlp_full


def get_nlp_lemma():
    global _nlp_lemma, _load_failed_lemma
    if _load_failed_lemma:
        return None
    if _nlp_lemma is not None:
        return _nlp_lemma
    with _lock:
        if _nlp_lemma is not None:
            return _nlp_lemma
        if _load_failed_lemma:
            return None
        try:
            _ensure_model_available()
            import spacy
            _nlp_lemma = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception as e:
            logger.warning(f"Failed to load spaCy lemma model: {e}")
            _load_failed_lemma = True
            return None
    return _nlp_lemma
