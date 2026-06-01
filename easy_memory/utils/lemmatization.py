"""
1. 将文本转小写后送入spaCy处理
2. 过滤掉标点符号和停用词（the, is, at等）
3. 取每个词的词形还原形式（lemma）：如 "running" -> "run", "dogs" -> "dog"
4. 对于以"ing"结尾的词，同时保留原词和词形还原形式，提高搜索召回率
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def lemmatize_for_bm25(text: str) -> str:
    from easy_memory.utils.spacy_models import get_nlp_lemma
    nlp = get_nlp_lemma()
    if nlp is None:
        return text
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_punct or token.is_stop:
            continue
        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)
    return " ".join(tokens)
