from abc import ABC

from transformers import GPT2Model, GPT2Config


class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *_, **__):
        return None


class AutoModel:
    @classmethod
    def from_pretrained(cls, *_, **__):
        class MockModel(GPT2Model, ABC):

            @classmethod
            def chat(cls, _, query, history) -> (str, list):
                response = 'Hello'
                return response, history + [[query, response]]

            @classmethod
            def stream_chat(cls, _, query, history) -> (str, list):
                from time import sleep
                response = ''
                for ch in 'Hello':
                    response += str(ch)
                    sleep(.2)
                    yield response, history + [[query, response]]

            def cuda(self, *args, **kwargs):
                return self

        return MockModel(GPT2Config())
