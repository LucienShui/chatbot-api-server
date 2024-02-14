from typing import Iterator
from copy import deepcopy

from util.openai_object import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Role
from .chatgpt import ChatGPT

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
}


class Qwen2VLLM(ChatGPT):
    def __init__(self, model: str, api_key: str, api_base: str, generation_config: dict = None):
        super().__init__(model=model, api_key=api_key, api_base=api_base)
        self.generation_config: dict = generation_config or DEFAULT_GENERATION_CONFIG

    def chat(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        request = deepcopy(request)
        request.temperature = request.temperature or self.generation_config['temperature']
        request.top_p = request.top_p or self.generation_config['top_p']
        if request.messages[0].role != Role.system:
            request.messages.insert(0, ChatMessage(role=Role.system, content='You are a helpful assistant.'))
        return super().chat(request)
