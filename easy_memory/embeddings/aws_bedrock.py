import json
import os
from typing import Literal, Optional
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class AWSBedrockEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.config.model = self.config.model or "amazon.titan-embed-text-v2:0"
        self.config.embedding_dims = self.config.embedding_dims or 1024
        import boto3
        kwargs = {"region_name": self.config.aws_region or os.getenv("AWS_REGION", "us-west-2")}
        if self.config.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.config.aws_access_key_id
        if self.config.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
        self.client = boto3.client("bedrock-runtime", **kwargs)

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        native_request = {"inputText": text}
        response = self.client.invoke_model(modelId=self.config.model, body=json.dumps(native_request))
        result = json.loads(response["body"].read())
        return result.get("embedding", [])
