from .base import ChatBotCompatible, Converter
from typing import List, Dict


class QwenChat(ChatBotCompatible):

    def __init__(self, pretrained: str):
        super(QwenChat, self).__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation.utils import GenerationConfig
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained, device_map='auto', trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(pretrained, trust_remote_code=True)

    def _chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        query, history = Converter.from_messages(messages)
        response, history = self.model.chat(self.tokenizer, query, history=history, **parameters)
        return response

    def _stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        query, history = Converter.from_messages(messages)
        for response in self.model.chat_stream(self.tokenizer, query, history=history, **parameters):
            yield response
