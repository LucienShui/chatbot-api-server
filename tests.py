import unittest


class RemoteCodeTestCase(unittest.TestCase):
    def test_remote_code(self):
        import os
        from model.base import from_config
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(tmp_dir)
            with open(os.path.join(tmp_dir, 'func.py'), 'w') as f:
                f.write("func = lambda x: x")
            with open(os.path.join(tmp_dir, 'model.py'), 'w') as f:
                f.write("""
from chatbot import ChatBotBase
from .func import func
class Bot(ChatBotBase):
    def __init__(self):
        super().__init__()
    def chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        return func(query)""")
            path = os.path.join(tmp_dir, 'model.Bot')
            obj = from_config(path, {})
            text = 'Hello World!'
            self.assertEqual(text, obj.chat(text))


class TestChatBot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from model.base import Converter
        from model.loader import from_bot_map_config
        from util import load_config
        config = load_config('config.json')
        cls.to_massages = Converter.to_messages
        cls.bot_map = from_bot_map_config(config['bot_map'])

    def test_gpt4_chat(self):
        print(self.bot_map['gpt-4'].chat(self.to_massages('你好')))
        print('=' * 16)
        for response in self.bot_map['gpt-4'].stream_chat(self.to_massages('你好')):
            print(response)

    def test_qwen_chat(self):
        from util.openai_object import ChatCompletionRequest, ChatMessage, Role
        model = 'qwen1.5-7b-chat-vllm'
        request = ChatCompletionRequest(model=model, messages=[
            ChatMessage(role=Role.system, content='You are a helpful assistant.'),
            ChatMessage(role=Role.user, content='Hi')
        ], temperature=0.05, top_p=0.8)
        # response: ChatCompletionResponse = next(self.bot_map[model].chat(request))
        # print(response.choices[0].message.content)

        print('=' * 16)
        request.stream = True
        for response in self.bot_map[model].chat(request):
            if content := response.choices[0].delta.content:
                print(content, end='', flush=True)

    def test_gpt3_chat(self):
        from util.openai_object import ChatCompletionRequest, ChatCompletionResponse
        model = 'gpt-3.5-turbo-azure'
        prompt = 'Hi'
        response: ChatCompletionResponse = next(self.bot_map[model].chat(
            ChatCompletionRequest(model=model, messages=self.to_massages(prompt))
        ))
        print(response.choices[0].message.content)

        print('=' * 16)
        for response in self.bot_map[model].chat(
                ChatCompletionRequest(model=model, messages=self.to_massages(prompt), stream=True)
        ):
            if content := response.choices[0].delta.content:
                print(content, end='', flush=True)

    def test_qwen2_chat(self):
        from util.openai_object import ChatCompletionRequest, ChatCompletionResponse
        model = 'mock-gpt'
        prompt = 'Hi'
        response: ChatCompletionResponse = next(self.bot_map[model].chat(
            ChatCompletionRequest(model=model, messages=self.to_massages(prompt))
        ))
        print(response.choices[0].message.content)

        print('=' * 16)
        for response in self.bot_map[model].chat(
                ChatCompletionRequest(model=model, messages=self.to_massages(prompt), stream=True)
        ):
            if content := response.choices[0].delta.content:
                print(content, end='', flush=True)
        self.assertTrue(True)


class ConverterTestCase(unittest.TestCase):
    def test_convertor(self):
        from model.base import Converter
        query = 'Hi'
        history = [['Hello', 'World!']]
        expected_messages = [
            {'role': Converter.user, 'content': 'Hello'},
            {'role': Converter.assistant, 'content': 'World!'},
            {'role': Converter.user, 'content': 'Hi'},
        ]
        converted_messages = Converter.to_messages(query, history)
        self.assertEqual(expected_messages, converted_messages)

        messages = [
            {'role': Converter.system, 'content': 'Hello'},
            {'role': Converter.user, 'content': 'Python!'},
            {'role': Converter.assistant, 'content': 'Hello'},
            {'role': Converter.assistant, 'content': 'World!'},
            {'role': Converter.user, 'content': 'Hi'},
            {'role': Converter.user, 'content': 'Python!'},
        ]

        expected_query = 'Hi\n\nPython!'
        expected_history = [['Hello\n\nPython!', 'Hello\n\nWorld!']]

        converted_query, converted_history = Converter.from_messages(messages)
        self.assertEqual(expected_query, converted_query)
        self.assertEqual(expected_history, converted_history)


class PydanticTestCase(unittest.TestCase):
    def test_pydantic(self):
        from util.openai_object import ChatCompletionUsage
        usage = ChatCompletionUsage(prompt_tokens=100, completion_tokens=100)
        print(usage.dict(exclude_unset=True))
        print(usage.json(exclude_unset=True))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
