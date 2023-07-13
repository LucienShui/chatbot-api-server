import os
import time
from functools import partial
from json import dumps
from typing import Dict, List, Tuple

import openai

from util import logger

dumps = partial(dumps, separators=(',', ':'), ensure_ascii=False)


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


class ChatBotBase:
    def __init__(self):
        self.logger = logger.getChild(self.__class__.__name__)

    def chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        raise NotImplementedError

    def stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        raise NotImplementedError


class BaichuanChat(ChatBotBase):
    def __init__(self, pretrained: str, quantize: int = None):
        super(BaichuanChat, self).__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation.utils import GenerationConfig
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained, torch_dtype=torch.float16, trust_remote_code=True)
        self.model = self.model.quantize(quantize).cuda() if quantize else self.model.cuda()
        self.model.generation_config = GenerationConfig.from_pretrained(pretrained)

    def chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        response = self.model.chat(self.tokenizer, messages, **parameters)
        return response

    def stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        for response in self.model.chat(self.tokenizer, messages, **parameters):
            yield response


class ChatGLM(ChatBotBase):

    def __init__(self, pretrained: str):
        from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
        super(ChatGLM, self).__init__()
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.model: PreTrainedModel = AutoModel.from_pretrained(pretrained, trust_remote_code=True, device='cuda')
        self.model = self.model.eval()

    def chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        query, history = Converter.from_messages(messages)
        response, history = self.model.chat(self.tokenizer, query, history=history, **parameters)
        return response

    def stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        query, history = Converter.from_messages(messages)
        for response, _ in self.model.stream_chat(self.tokenizer, query, history=history, **parameters):
            yield response


class ChatGPT(ChatBotBase):

    def __init__(self, model: str, api_key: str, api_base: str = 'https://api.openai-proxy.com/v1'):
        super().__init__()
        self.model: str = model
        self.api_base: str = api_base
        self.api_key: str = api_key

    def chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        start_time = time.time()
        openai_response = openai.ChatCompletion.create(model=self.model, messages=messages, api_base=self.api_base,
                                                       api_key=self.api_key, **parameters or {})
        content: str = openai_response['choices'][0]['message']['content']
        self.logger.info(dumps({'messages': messages, 'parameters': parameters,
                                'response': content, 'cost': f'{(time.time() - start_time) * 1000:.2f} ms'}))
        return content

    def stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
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

            self.logger.info(dumps({'messages': messages, 'parameters': parameters,
                                    'response': message, 'cost': f'{(time.time() - start_time) * 1000:.2f} ms'}))


class SpecialChatGPT(ChatGPT):
    def chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        response = ''
        for response in self.stream_chat(messages, parameters):
            pass
        return response


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


supported_class = {c.__name__: c for c in [ChatGPT, AzureChatGPT, BaichuanChat, ChatGLM, SpecialChatGPT]}


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
