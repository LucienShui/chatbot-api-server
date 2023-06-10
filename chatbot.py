from requests.api import post
import time
from urllib.parse import urljoin
from typing import Dict, List, Any
from util import logger
from json import dumps
from functools import partial

dumps = partial(dumps, separators=(',', ':'), ensure_ascii=False)


class ChatBotBase:
    def __init__(self):
        self.logger = logger.getChild(self.__class__.__name__)

    def chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        raise NotImplementedError

    def stream_chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        raise NotImplementedError


class ChatGLM(ChatBotBase):
    def __init__(self, pretrained: str):
        from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
        super().__init__()
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.model: PreTrainedModel = AutoModel.from_pretrained(pretrained, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        response, history = self.model.chat(self.tokenizer, query, history=history, **parameters)
        return response


class ChatGPTBase(ChatBotBase):

    def __init__(self, url: str, headers: Dict[str, str]):
        super().__init__()
        self.url: str = url
        self.headers: Dict[str, str] = headers

    def make_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        raise NotImplementedError

    def chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        history = history or []
        messages: list = [{'role': 'system', 'content': system}] if system else []
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": query})
        request: dict = self.make_request(messages)
        start_time = time.time()
        response: dict = post(self.url, json=request, headers=self.headers).json()
        duration = time.time() - start_time
        log_msg = f"request = {dumps(request)}, response = {dumps(response)}, cost = {round(duration * 1000, 2)} ms"
        self.logger.info(log_msg)
        if 'error' in response:
            message: str = f"Error: {response['error']['message']}"
        else:
            message: str = response['choices'].pop(0)['message']['content']
        return message


class ChatGPT(ChatGPTBase):
    def __init__(self, api_key: str, endpoint: str = 'https://api.openai-proxy.com',
                 model: str = "gpt-3.5-turbo", organization: str = None):
        assert model in ["gpt-4-0314", "gpt-4", "gpt-3.5-turbo-0301", "gpt-3.5-turbo"]
        self.model: str = model
        headers = {'Authorization': f'Bearer {api_key}', 'OpenAI-Organization': organization}
        super().__init__(url=urljoin(endpoint, '/v1/chat/completions'), headers=headers)

    def make_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return {"model": self.model, "messages": messages}


class AzureChatGPT(ChatGPTBase):
    def __init__(self, api_key: str, endpoint: str):
        super().__init__(endpoint, {'api-key': api_key})

    def make_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return {"messages": messages}


class ChatRemote(ChatBotBase):
    def __init__(self, url: str, preset_history: list = None):
        super().__init__()
        self.url = url
        self.preset_history = preset_history or []

    def chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        history = self.preset_history + (history or [])
        request: dict = {"query": query, "history": history, "system": system}
        start_time = time.time()
        response: dict = post(self.url, json=request).json()
        duration = time.time() - start_time
        log_msg = f"request = {dumps(request)}, response = {dumps(response)}, cost = {round(duration * 1000, 2)} ms"
        self.logger.info(log_msg)
        message: str = response['response']
        return message


def from_config(bot_class: str, config: dict) -> ChatBotBase:
    supported_class = ['ChatGPT', 'AzureChatGPT', 'ChatRemote']
    assert bot_class in supported_class
    if bot_class == 'ChatGPT':
        return ChatGPT(**config)
    if bot_class == 'AzureChatGPT':
        return AzureChatGPT(**config)
    if bot_class == 'ChatRemote':
        return ChatRemote(**config)
    raise ModuleNotFoundError(bot_class)


def from_bot_map_config(bot_map_config: Dict[str, dict]) -> Dict[str, ChatBotBase]:
    bot_map: Dict[str, ChatBotBase] = {}
    for bot, config in bot_map_config.items():
        bot_class = config.pop('class')
        bot_map[bot] = from_config(bot_class, config)
    return bot_map
