import os
import time
from functools import partial
from json import dumps
from typing import Dict, List

import openai
from requests.api import post

from util import logger

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

    def stream_chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        for response, _ in self.model.stream_chat(self.tokenizer, query, history=history, **parameters):
            yield response


class ChatGPT(ChatBotBase):

    def __init__(self, model: str, api_key: str, api_base: str = 'https://api.openai-proxy.com/v1'):
        super().__init__()
        self.model: str = model
        self.api_base: str = api_base
        self.api_key: str = api_key

    @classmethod
    def get_messages(cls, query: str, history: list = None, system: str = None) -> List[Dict[str, str]]:
        history = history or []
        messages: list = [{'role': 'system', 'content': system}] if system else []
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": query})
        return messages

    def chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        messages: list = self.get_messages(query, history, system)
        start_time = time.time()
        response = openai.ChatCompletion.create(model=self.model, messages=messages, api_base=self.api_base,
                                                api_key=self.api_key, **parameters or {})
        message: str = response['choices'][0]['message']['content']
        self.logger.info(dumps({'query': query, 'history': history, 'system': system, 'parameters': parameters,
                                'response': message, 'cost': f'{(time.time() - start_time) * 1000:.2f} ms'}))
        return message

    def stream_chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        messages = self.get_messages(query, history, system)
        start_time = time.time()
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

        self.logger.info(dumps({'query': query, 'history': history, 'system': system, 'parameters': parameters,
                                'response': message, 'cost': f'{(time.time() - start_time) * 1000:.2f} ms'}))


class SpecialChatGPT(ChatGPT):
    def chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        message = ''
        for message in self.stream_chat(query, history, system, parameters):
            pass
        return message


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

    def chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        return super().chat(query, history, system, self.parameters_wrapper(parameters))

    def stream_chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        return super().stream_chat(query, history, system, self.parameters_wrapper(parameters))


class ChatRemote(ChatBotBase):
    def __init__(self, url: str, preset_history: list = None):
        super().__init__()
        self.url = url
        self.preset_history = preset_history or []

    def chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        history = self.preset_history + (history or [])
        request: dict = {"query": query, "history": history, "system": system, "parameters": parameters}
        start_time = time.time()
        response: dict = post(self.url, json=request).json()
        duration = time.time() - start_time
        log_msg = f"request = {dumps(request)}, response = {dumps(response)}, cost = {round(duration * 1000, 2)} ms"
        self.logger.info(log_msg)
        message: str = response['response']
        return message


supported_class = {c.__name__: c for c in [ChatGPT, AzureChatGPT, ChatRemote, ChatGLM, SpecialChatGPT]}


def import_remote(module_path: str, config: dict):
    """
    Args:
        module_path:
            a path like "/tmp/test_model/model.Bot"
            which model means model.py, Bot is a class's name inside model.py
        config:
            module's __init__ parameters
    Return:
        an object of module
    """
    import sys
    import tempfile
    import shutil
    package_name: str = 'remote_code'
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir, module = os.path.split(module_path)
        shutil.copytree(base_dir, os.path.join(tmp_dir, package_name))
        filename, class_name = os.path.splitext(module)
        class_name = class_name.replace('.', '')

        dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # do not generate __pycache__ inside directory
        sys.path.insert(0, tmp_dir)

        exec(f'from {package_name}.{filename} import {class_name}')
        obj = eval(class_name)(**config)
        sys.path.pop(0)
        sys.dont_write_bytecode = dont_write_bytecode
        return obj


def from_config(bot_class: str, config: dict) -> ChatBotBase:
    if bot_class in supported_class.keys():
        return supported_class[bot_class](**config)
    else:
        try:
            return import_remote(bot_class, config)
        except Exception as e:
            logger.exception(e)
            raise ModuleNotFoundError(f"{bot_class} not in {list(supported_class.keys())}")


def from_bot_map_config(bot_map_config: Dict[str, dict]) -> Dict[str, ChatBotBase]:
    bot_map: Dict[str, ChatBotBase] = {}
    for bot, config in bot_map_config.items():
        bot_class = config.pop('class')
        bot_map[bot] = from_config(bot_class, config)
    return bot_map
