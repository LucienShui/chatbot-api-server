from typing import Dict, List, Tuple, Iterator

from util.logger import logger
from util.openai_object import (
    ChatCompletionResponse, ChatCompletionRequest,
    ChatCompletionUsage
)


class Converter:
    system: str = 'system'
    user: str = 'user'
    assistant: str = 'assistant'

    @classmethod
    def to_messages(cls, query: str, history: list = None, system: str = None) -> List[Dict[str, str]]:
        history = history or []
        messages: list = [{'role': cls.system, 'content': system}] if system else []
        for q, a in history:
            messages.append({"role": cls.user, "content": q})
            messages.append({"role": cls.assistant, "content": a})
        messages.append({"role": cls.user, "content": query})
        return messages

    @classmethod
    def from_messages(cls, messages: List[Dict[str, str]]) -> Tuple[str, List[List[str]]]:
        line_breaker = '\n\n'
        assert messages[-1]['role'] == cls.user, 'last query must from user for ChatGLM'
        query = messages[-1]['content']
        history: List[List[str]] = []
        for message in messages[:-1]:
            role = message['role']
            content = message['content']
            if role in [cls.user, cls.system]:
                if len(history) == 0 or len(history[-1]) == 2:
                    history.append([content])
                else:
                    history[-1][0] += line_breaker + content
            elif role in [cls.assistant]:
                if len(history) == 0:
                    raise AssertionError('first query must from user or system for ChatGLM')
                if len(history[-1]) == 1:
                    history[-1].append(content)
                else:
                    history[-1][1] += line_breaker + content

        if len(history) > 0 and len(history[-1]) == 1:
            query = history.pop(-1)[0] + line_breaker + query

        return query, history


class ChatAPIBase:
    def __init__(self):
        self.logger = logger.getChild(self.__class__.__name__)

    def chat(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        raise NotImplementedError


class ChatAPICompatible(ChatAPIBase):

    def _chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        raise NotImplementedError

    def _stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        raise NotImplementedError

    def chat(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        parameters: dict = {}
        for k in ['temperature', 'top_p']:
            v = getattr(request, k)
            if v is not None:
                parameters[k] = v
        if request.max_tokens is not None:
            parameters['max_length'] = request.max_tokens
        messages = [message.dict() for message in request.messages]
        if request.stream:
            yield request.response()
            current_length = 0

            for new_response in self._stream_chat(messages, parameters=parameters):
                if len(new_response) == current_length:
                    continue

                new_text = new_response[current_length:]
                current_length = len(new_response)
                yield request.response(content=new_text)
            yield request.response(usage=ChatCompletionUsage())
        else:
            response = self._chat(messages, parameters=parameters)
            yield request.response(content=response, usage=ChatCompletionUsage())
