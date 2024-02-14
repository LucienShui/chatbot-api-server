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
        param = {
            **request.dict(exclude_unset=True),
            **{'model': self.model, 'api_base': self.api_base, 'api_key': self.api_key}
        }
        prompt_tokens, completion_tokens, total_tokens = [None] * 3
        if request.stream:
            yield request.response()
            blank_buffer = ''  # 有些时候模型会在末尾疯狂输出 \n，在此做一下判断
            for chunk in openai.ChatCompletion.create(**param):
                if not chunk['choices']:
                    continue
                if chunk['choices'][0]['finish_reason'] is not None:
                    prompt_tokens = chunk.get('usage', {}).get('prompt_tokens', None)
                    completion_tokens = chunk.get('usage', {}).get('completion_tokens', None)
                    total_tokens = chunk.get('usage', {}).get('total_tokens', None)
                    break
                if 'content' not in (delta := chunk['choices'][0]['delta']):
                    continue
                if not (delta_content := delta['content']):
                    continue
                if delta_content.strip() == '':
                    blank_buffer += delta_content
                    continue
                if blank_buffer:
                    delta_content = blank_buffer + delta_content
                    blank_buffer = ''
                yield request.response(content=delta_content)
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            yield request.response(usage=usage)

        else:
            openai_response = openai.ChatCompletion.create(**param)
            content: str = openai_response['choices'][0]['message']['content']
            prompt_tokens = openai_response['usage']['prompt_tokens']
            completion_tokens = openai_response['usage']['completion_tokens']
            total_tokens = openai_response['usage']['total_tokens']
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            content = content.strip()  # 有些时候模型会在末尾疯狂输出 \n，在此做一下判断
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
