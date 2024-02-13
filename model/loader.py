import os
from typing import Dict, List

from util.logger import logger
from .baichuan import Baichuan
from .base import ChatAPIBase
from .chatglm import ChatGLM, ChatGLM3
from .chatgpt import ChatGPT, AzureChatGPT
from .qwen import Qwen, Qwen2
from .mock import Mock

supported_class = {c.__name__: c for c in [ChatGPT, AzureChatGPT, Baichuan, ChatGLM, ChatGLM3, Qwen, Qwen2, Mock]}


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


def from_config(bot_class: str, config: dict) -> ChatAPIBase:
    if bot_class in supported_class.keys():
        return supported_class[bot_class](**config)
    else:
        try:
            return import_remote(bot_class, config)
        except Exception as e:
            logger.exception(e)
            raise ModuleNotFoundError(f"{bot_class} not in {list(supported_class.keys())}")


def check_alias(alias: Dict[str, str], config: Dict[str, Dict[str, str]]) -> bool:
    link_count: Dict[str, int] = {}
    for link, source in alias.items():
        assert source in config, f"illegal alias: {source} not found in bot_map"
        link_count[link] = link_count.get(link, 0) + 1
    for link, count in link_count.items():
        if count > 1:
            raise AssertionError(f"duplicated alias: {link}")
    return True


def from_bot_map_config(bot_map_config: Dict[str, dict], alias: Dict[str, str] = None,
                        disable: List[str] = None) -> Dict[str, ChatAPIBase]:
    alias = alias or {}
    disable = disable or []
    map_config = {k: v for k, v in bot_map_config.items() if k not in disable}
    check_alias(alias, map_config)
    bot_map: Dict[str, ChatAPIBase] = {}
    for bot, config in map_config.items():
        bot_class = config.pop('class')
        bot_map[bot] = from_config(bot_class, config)
    for link, source in alias.items():
        bot_map[link] = bot_map[source]
    return bot_map
