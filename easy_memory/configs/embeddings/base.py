import os
from abc import ABC
from typing import Dict, Optional, Union

import httpx

from easy_memory.configs.base import AzureConfig


class BaseEmbedderConfig(ABC):
    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            embedding_dims: Optional[int] = None,
            ollama_base_url: Optional[str] = None,
            openai_base_url: Optional[str] = None,
            model_kwargs: Optional[dict] = None,
            huggingface_base_url: Optional[str] = None,
            azure_kwargs: Optional[AzureConfig] = {},
            http_client_proxies: Optional[Union[Dict, str]] = None,
            vertex_credentials_json: Optional[str] = None,
            memory_add_embedding_type: Optional[str] = None,
            memory_update_embedding_type: Optional[str] = None,
            memory_search_embedding_type: Optional[str] = None,
            output_dimensionality: Optional[str] = None,
            lmstudio_base_url: Optional[str] = "http://localhost:1234/v1",
            aws_access_key_id: Optional[str] = None,
            aws_secret_access_key: Optional[str] = None,
            aws_region: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.openai_base_url = openai_base_url
        self.embedding_dims = embedding_dims
        self.http_client = httpx.Client(proxies=http_client_proxies) if http_client_proxies else None
        self.ollama_base_url = ollama_base_url
        self.model_kwargs = model_kwargs or {}
        self.huggingface_base_url = huggingface_base_url
        self.azure_kwargs = AzureConfig(**azure_kwargs) if azure_kwargs else {}
        self.vertex_credentials_json = vertex_credentials_json
        self.memory_add_embedding_type = memory_add_embedding_type
        self.memory_update_embedding_type = memory_update_embedding_type
        self.memory_search_embedding_type = memory_search_embedding_type
        self.output_dimensionality = output_dimensionality
        self.lmstudio_base_url = lmstudio_base_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region or os.environ.get("AWS_REGION") or "us-west-2"
