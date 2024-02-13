from typing import Iterator
import time

from util.openai_object import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage
from .base import ChatAPIBase

DEFAULT_RESPONSE = 'Hi，这是一句用于测试的回答。\nHi, this is a mock response for testing.'


class Mock(ChatAPIBase):

    def __init__(self, response: str = DEFAULT_RESPONSE, sleep: float = 0.02):
        super().__init__()
        self.response: str = response
        self.sleep: float = sleep

    def chat(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        prompt_tokens: int = sum(map(lambda x: len(x.content), request.messages))
        completion_tokens: int = len(self.response)
        usage = ChatCompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                    total_tokens=prompt_tokens + completion_tokens)
        if request.stream:
            yield request.response()
            for i, delta in enumerate(self.response):
                yield request.response(content=delta)
                time.sleep(self.sleep)
            yield request.response(usage=usage)
        else:
            yield request.response(content=self.response, usage=usage)
