from copy import deepcopy
from threading import Thread
from typing import List, Dict, Iterator

from transformers import TextIteratorStreamer

from util.openai_object import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatCompletionUsage, Role
from .base import ChatAPICompatible, Converter, ChatAPIBase


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
        import json
        import os
        self.model = AutoModelForCausalLM.from_pretrained(pretrained, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        with open(os.path.join(pretrained, 'generation_config.json'), 'r') as f:
            self.generation_config = json.load(f)
        self.im_end = '<|im_end|>'

    def chat(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionResponse]:
        # 1. process input parameters
        messages = request.messages
        if messages[0].role != Role.system:
            messages = [ChatMessage(role=Role.system, content='You are a helpful assistant.')] + messages
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        prompt_tokens: int = len(model_inputs[0])

        parameters: dict = {}
        for k in ['temperature', 'top_p']:
            v = getattr(request, k)
            if v is not None:
                parameters[k] = v

        parameters = parameters or deepcopy(self.generation_config)

        parameters['max_new_tokens'] = request.max_tokens or 2048
        parameters['pad_token_id'] = self.tokenizer.eos_token_id

        if request.stream:
            yield request.response()

            # https://huggingface.co/docs/transformers/v4.37.2/en/internal/generation_utils#transformers.TextIteratorStreamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            generation_kwargs = {**model_inputs, "streamer": streamer, **parameters}
            # 2. call generate
            Thread(target=self.model.generate, kwargs=generation_kwargs).start()
            response = ""
            drop_flag = False
            for delta in streamer:
                response += delta
                if not drop_flag and delta:
                    if self.im_end in delta:  # 如果 delta 中包含了<|im_end|>，则说明模型的本轮回答已经结束
                        drop_flag = True
                        delta = delta[:delta.index(self.im_end)]
                    yield request.response(content=delta)

            # 3. hf's streamer return text directly, to count the token we need to encode it again.
            output_ids: List[int] = self.tokenizer.encode(response)
            completion_tokens = len(output_ids)
            total_tokens = prompt_tokens + completion_tokens
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            yield request.response(usage=usage)
        else:
            # 2. call generate
            generated_ids_batch = self.model.generate(model_inputs.input_ids, **parameters)
            total_tokens: int = len(generated_ids_batch[0])
            # 3. convert batch output to text
            output_ids_batch = [output_ids[len(input_ids):] for input_ids, output_ids in
                                zip(model_inputs.input_ids, generated_ids_batch)]

            completion_tokens: int = len(output_ids_batch[0])
            assert prompt_tokens + completion_tokens == total_tokens
            response = self.tokenizer.batch_decode(output_ids_batch, skip_special_tokens=True)[0]
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            yield request.response(content=response, usage=usage)
