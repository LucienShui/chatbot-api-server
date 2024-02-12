from .base import ChatBotBase
from typing import List, Dict
import openai


class ChatGPT(ChatBotBase):

    def __init__(self, model: str, api_key: str, api_base: str = 'https://api.openai-proxy.com/v1'):
        super().__init__()
        self.model: str = model
        self.api_base: str = api_base
        self.api_key: str = api_key

    def chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        openai_response = openai.ChatCompletion.create(model=self.model, messages=messages, api_base=self.api_base,
                                                       api_key=self.api_key, **parameters or {})
        content: str = openai_response['choices'][0]['message']['content']
        return content

    def stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
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


class AzureChatGPT(ChatGPT):
    def __init__(self, engine: str, api_key: str, api_base: str, api_version: str):
        super().__init__(engine, api_key, api_base)
        self.api_version: str = api_version
        self.api_type: str = 'azure'

    def parameters_wrapper(self, parameters: dict) -> dict:
        parameters: dict = parameters or {}
        parameters['api_type'] = self.api_type
        parameters['api_version'] = self.api_version
        parameters['engine'] = self.model
        return parameters

    def chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        return super().chat(messages, self.parameters_wrapper(parameters))

    def stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        return super().stream_chat(messages, self.parameters_wrapper(parameters))
