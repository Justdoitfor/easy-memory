import hashlib
import logging
import re
from typing import Any, Dict, List

from easy_memory.configs.prompts import (
    AGENT_MEMORY_EXTRACTION_PROMPT,
    FACT_RETRIEVAL_PROMPT,
    USER_MEMORY_EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)


def get_fact_retrieval_messages(message, is_agent_memory=False):
    if is_agent_memory:
        return AGENT_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"
    else:
        return USER_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"


def get_fact_retrieval_messages_legacy(message):
    return FACT_RETRIEVAL_PROMPT, f"Input:\n{message}"


def ensure_json_instruction(system_prompt, user_prompt):
    combined = (system_prompt + user_prompt).lower()
    if "json" not in combined:
        system_prompt += "\n\nYou must return your response in valid JSON format with a 'facts' key containing an array of strings."
    return system_prompt, user_prompt


def parse_messages(messages):
    response = ""
    for msg in messages:
        if msg["role"] == "system":
            response += f"system: {msg['content']}\n"
        if msg["role"] == "user":
            response += f"user: {msg['content']}\n"
        if msg["role"] == "assistant":
            response += f"assistant: {msg['content']}\n"
    return response


def format_entities(entities):
    if not entities:
        return ""
    formatted_lines = []
    for entity in entities:
        simplified = f"{entity['source']} -- {entity['relationship']} -- {entity['destination']}"
        formatted_lines.append(simplified)
    return "\n".join(formatted_lines)


def normalize_facts(raw_facts):
    if not raw_facts:
        return []
    normalized = []
    for item in raw_facts:
        if isinstance(item, str):
            fact = item
        elif isinstance(item, dict):
            fact = item.get("fact") or item.get("text")
            if fact is None:
                continue
        else:
            fact = str(item)
        if fact:
            normalized.append(fact)
    return normalized


def remove_code_blocks(content: str) -> str:
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, content.strip())
    match_res = match.group(1).strip() if match else content.strip()
    return re.sub(r"<think>.*?</think>", "", match_res, flags=re.DOTALL).strip()


def extract_json(text):
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx: end_idx + 1]
        else:
            json_str = text
    return json_str


def get_image_description(image_obj, llm, vision_details):
    if isinstance(image_obj, str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "A user is providing an image. Provide a high level description of the image and do not include any additional text."},
                    {"type": "image_url", "image_url": {"url": image_obj, "detail": vision_details}},
                ],
            },
        ]
    else:
        messages = [image_obj]
    return llm.generate_response(messages=messages)


def parse_vision_messages(messages, llm=None, vision_details="auto"):
    returned_messages = []
    for msg in messages:
        if msg["role"] == "system":
            returned_messages.append(msg)
            continue
        if isinstance(msg["content"], list):
            description = get_image_description(msg, llm, vision_details)
            returned_messages.append({"role": msg["role"], "content": description})
        elif isinstance(msg["content"], dict) and msg["content"].get("type") == "image_url":
            image_url = msg["content"]["image_url"]["url"]
            try:
                description = get_image_description(image_url, llm, vision_details)
                returned_messages.append({"role": msg["role"], "content": description})
            except Exception:
                raise Exception(f"Error while downloading {image_url}.")
        else:
            returned_messages.append(msg)
    return returned_messages


def process_telemetry_filters(filters):
    if filters is None:
        return {}
    encoded_ids = {}
    if "user_id" in filters:
        encoded_ids["user_id"] = hashlib.md5(filters["user_id"].encode()).hexdigest()
    if "agent_id" in filters:
        encoded_ids["agent_id"] = hashlib.md5(filters["agent_id"].encode()).hexdigest()
    if "run_id" in filters:
        encoded_ids["run_id"] = hashlib.md5(filters["run_id"].encode()).hexdigest()
    return list(filters.keys()), encoded_ids


def sanitize_relationship_for_cypher(relationship) -> str:
    char_map = {
        "...": "_ellipsis_", "…": "_ellipsis_", "。": "_period_", "，": "_comma_",
        "；": "_semicolon_", "：": "_colon_", "！": "_exclamation_", "？": "_question_",
        "（": "_lparen_", "）": "_rparen_", "【": "_lbracket_", "】": "_rbracket_",
        "《": "_langle_", "》": "_rangle_", "'": "_apostrophe_", '"': "_quote_",
        "\\": "_backslash_", "/": "_slash_", "|": "_pipe_", "&": "_ampersand_",
        "=": "_equals_", "+": "_plus_", "*": "_asterisk_", "^": "_caret_",
        "%": "_percent_", "$": "_dollar_", "#": "_hash_", "@": "_at_",
        "!": "_bang_", "?": "_question_", "(": "_lparen_", ")": "_rparen_",
        "[": "_lbracket_", "]": "_rbracket_", "{": "_lbrace_", "}": "_rbrace_",
        "<": "_langle_", ">": "_rangle_", "-": "_",
    }
    sanitized = relationship
    for old, new in char_map.items():
        sanitized = sanitized.replace(old, new)
    return re.sub(r"_+", "_", sanitized).strip("_")


def remove_spaces_from_entities(entity_list, *, sanitize_relationship=True):
    required = ("source", "relationship", "destination")
    cleaned = []
    for item in entity_list:
        if not isinstance(item, dict) or not item:
            continue
        if not all(key in item for key in required):
            continue
        item["source"] = item["source"].lower().replace(" ", "_")
        rel = item["relationship"].lower().replace(" ", "_")
        item["relationship"] = sanitize_relationship_for_cypher(rel) if sanitize_relationship else rel
        item["destination"] = item["destination"].lower().replace(" ", "_")
        cleaned.append(item)
    return cleaned
