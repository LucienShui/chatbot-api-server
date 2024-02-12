from util.openai_object import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseStreamChoice,
    DeltaMessage, ChatCompletionResponseChoice, ChatMessage, ChatCompletionUsage
)
from .base import ChatAPICompatible, ChatAPIBase
from typing import List, Dict, Iterator
import openai


class ChatGPTBak(ChatAPICompatible):

    def __init__(self, model: str, api_key: str, api_base: str = 'https://api.openai-proxy.com/v1'):
        super().__init__()
        self.model: str = model
        self.api_base: str = api_base
        self.api_key: str = api_key

    def _chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        openai_response = openai.ChatCompletion.create(model=self.model, messages=messages, api_base=self.api_base,
                                                       api_key=self.api_key, **parameters or {})
        content: str = openai_response['choices'][0]['message']['content']
        prompt_tokens = openai_response['usage']['prompt_tokens']
        completion_tokens = openai_response['usage']['completion_tokens']
        total_tokens = openai_response['usage']['total_tokens']
        return content

    def _stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        response = openai.ChatCompletion.create(model=self.model, messages=messages, api_base=self.api_base,
                                                api_key=self.api_key, stream=True, **parameters or {})
        message = ''
        for chunk in response:
            if chunk['choices'][0]['finish_reason'] is not None:
                break
            if 'content' not in (delta := chunk['choices'][0]['delta']):
                continue
            delta_content = delta['content']
            message += delta_content
            yield message


class ChatGPT(ChatAPIBase):

    def __init__(self, model: str, api_key: str, api_base: str = 'https://api.openai-proxy.com/v1'):
        super().__init__()
        self.model: str = model
        self.api_base: str = api_base
        self.api_key: str = api_key

    def chat(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        param = {**request.dict(), **{'api_base': self.api_base, 'api_key': self.api_key}}
        if request.stream:
            completion_tokens = 0
            choice = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None
            )
            yield ChatCompletionResponse(model=request.model, choices=[choice], object="chat.completion.chunk")
            for chunk in openai.ChatCompletion.create(**param):
                if not chunk['choices']:
                    continue
                if chunk['choices'][0]['finish_reason'] is not None:
                    break
                if 'content' not in (delta := chunk['choices'][0]['delta']):
                    continue
                if not (delta_content := delta['content']):
                    continue
                completion_tokens += 1
                choice = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=delta_content),
                    finish_reason=None
                )
                yield ChatCompletionResponse(model=request.model, choices=[choice], object="chat.completion.chunk")
            choice = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop"
            )
            yield ChatCompletionResponse(model=request.model, choices=[choice],
                                         usage=ChatCompletionUsage(completion_tokens=completion_tokens),
                                         object="chat.completion.chunk")

        else:
            openai_response = openai.ChatCompletion.create(**param)
            content: str = openai_response['choices'][0]['message']['content']
            prompt_tokens = openai_response['usage']['prompt_tokens']
            completion_tokens = openai_response['usage']['completion_tokens']
            total_tokens = openai_response['usage']['total_tokens']
            choice = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop"
            )
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            yield ChatCompletionResponse(model=request.model, choices=[choice], object="chat.completion", usage=usage)


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
