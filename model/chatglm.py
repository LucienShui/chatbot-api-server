from .base import ChatAPICompatible, Converter
from typing import List, Dict, Tuple


class ChatGLM(ChatAPICompatible):

    def __init__(self, pretrained: str):
        from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
        super(ChatGLM, self).__init__()
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.model: PreTrainedModel = AutoModel.from_pretrained(pretrained, trust_remote_code=True).cuda()
        self.model = self.model.eval()

    def _chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        query, history = Converter.from_messages(messages)
        response, history = self.model.chat(self.tokenizer, query, history=history, **parameters)
        return response

    def _stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        query, history = Converter.from_messages(messages)
        for response, _ in self.model.stream_chat(self.tokenizer, query, history=history, **parameters):
            yield response


class ChatGLM3(ChatGLM):

    @classmethod
    def export_query(cls, messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        last_message = messages.pop(-1)
        assert last_message['role'] == 'user', f"last message's role should be user, got {last_message['role']}"
        return last_message['content'], messages

    def _chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        query, history = self.export_query(messages)
        response, history = self.model.chat(self.tokenizer, query, history=history, **parameters)
        return response

    def _stream_chat(self, messages: List[Dict[str, str]], parameters: dict = None) -> str:
        query, history = self.export_query(messages)
        for response, _ in self.model.stream_chat(self.tokenizer, query, history=history, **parameters):
            yield response
