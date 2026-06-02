import json
import os
from typing import Dict, List, Optional, Union

from easy_memory.configs.llms.aws_bedrock import AWSBedrockConfig
from easy_memory.configs.llms.base import BaseLlmConfig
from easy_memory.llms.base import LLMBase


class AWSBedrockLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, AWSBedrockConfig, Dict]] = None):
        if config is None:
            config = AWSBedrockConfig()
        elif isinstance(config, dict):
            config = AWSBedrockConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, AWSBedrockConfig):
            config = AWSBedrockConfig(
                model=config.model, temperature=config.temperature, api_key=config.api_key,
                max_tokens=config.max_tokens, top_p=config.top_p, top_k=config.top_k,
            )
        super().__init__(config)

        import boto3

        aws_config = self.config.get_aws_config()
        self.client = boto3.client("bedrock-runtime", **aws_config)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        model_id = self.config.model
        provider = self.config.provider

        if provider == "anthropic":
            system_msg = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)

            native_request = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": user_messages,
            }
            if system_msg:
                native_request["system"] = system_msg
        else:
            user_messages = [msg for msg in messages if msg["role"] != "system"]
            native_request = {
                "inputText": "\n".join(m["content"] for m in user_messages),
                "textGenerationConfig": {
                    "maxTokenCount": self.config.max_tokens,
                    "temperature": self.config.temperature,
                },
            }

        response = self.client.invoke_model(
            modelId=model_id, body=json.dumps(native_request)
        )
        result = json.loads(response["body"].read())

        if provider == "anthropic":
            return result.get("content", [{}])[0].get("text", "")
        return result.get("results", [{}])[0].get("outputText", "")