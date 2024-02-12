from .base import ChatAPICompatible, Converter
from typing import List, Dict


class Baichuan(ChatAPICompatible):
    def __init__(self, pretrained: str, quantize: int = None):
        super(Baichuan, self).__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained, torch_dtype=torch.float16, trust_remote_code=True)
        self.model = self.model.quantize(quantize).cuda() if quantize else self.model.cuda()
        self.model = self.model.eval()
        self.model.generation_config = GenerationConfig.from_pretrained(pretrained)

    def to_generation_config(self, parameters: dict = None):
        from transformers.generation import GenerationConfig
        generation_config = GenerationConfig.from_dict({**self.model.generation_config.to_dict(), **(parameters or {})})
        return generation_config

    @classmethod
    def process_messages(cls, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        result = []
        for message in messages:
            if message['role'] == Converter.system:
                result.append({'role': Converter.user, 'content': message['content']})
            else:
                result.append(message)
        return result

    def _chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        response = self.model.chat(
            self.tokenizer, self.process_messages(messages), generation_config=self.to_generation_config(parameters))
        return response

    def _stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        for response in self.model.chat(self.tokenizer, self.process_messages(messages), stream=True,
                                        generation_config=self.to_generation_config(parameters)):
            yield response
