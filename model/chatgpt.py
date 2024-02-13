from typing import Iterator

import openai

from util.openai_object import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage
from .base import ChatAPIBase


class ChatGPT(ChatAPIBase):

    def __init__(self, model: str, api_key: str, api_base: str = 'https://api.openai-proxy.com/v1'):
        super().__init__()
        self.model: str = model
        self.api_base: str = api_base
        self.api_key: str = api_key

    def chat(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        param = {**request.dict(), **{'api_base': self.api_base, 'api_key': self.api_key}}
        if request.stream:
            yield request.response()
            for chunk in openai.ChatCompletion.create(**param):
                if not chunk['choices']:
                    continue
                if chunk['choices'][0]['finish_reason'] is not None:
                    break
                if 'content' not in (delta := chunk['choices'][0]['delta']):
                    continue
                if not (delta_content := delta['content']):
                    continue
                yield request.response(content=delta_content)
            yield request.response(usage=ChatCompletionUsage())

        else:
            openai_response = openai.ChatCompletion.create(**param)
            content: str = openai_response['choices'][0]['message']['content']
            prompt_tokens = openai_response['usage']['prompt_tokens']
            completion_tokens = openai_response['usage']['completion_tokens']
            total_tokens = openai_response['usage']['total_tokens']
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            yield request.response(content=content, usage=usage)


class AzureChatGPT(ChatGPT):
    def __init__(self, engine: str, api_key: str, api_base: str, api_version: str):
        super().__init__(engine, api_key, api_base)
        self.api_version: str = api_version
        self.api_type: str = 'azure'

    def chat(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        request.api_type = self.api_type
        request.api_version = self.api_version
        request.engine = self.model
        return super().chat(request)
