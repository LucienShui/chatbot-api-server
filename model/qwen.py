from util.openai_object import (ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice,
                                ChatMessage, ChatCompletionUsage)
from .base import ChatAPICompatible, Converter, ChatAPIBase
from typing import List, Dict, Iterator


class Qwen(ChatAPICompatible):

    def __init__(self, pretrained: str):
        super(Qwen, self).__init__()
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


class Qwen2(ChatAPIBase):
    def __init__(self, pretrained: str):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(pretrained, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def chat(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        messages = request.messages
        if messages[0].role != 'system':
            request.messages.insert(0, ChatMessage(role='system', content='You are a helpful assistant.'))
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        prompt_tokens: int = len(model_inputs[0])

        if request.stream and False:
            pass
        else:
            generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=request.max_tokens)
            total_tokens: int = len(generated_ids[0])
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                             zip(model_inputs.input_ids, generated_ids)]

            completion_tokens: int = len(generated_ids[0])
            assert prompt_tokens + completion_tokens == total_tokens
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            choice = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop"
            )
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            yield ChatCompletionResponse(model=request.model, choices=[choice], object="chat.completion", usage=usage)
