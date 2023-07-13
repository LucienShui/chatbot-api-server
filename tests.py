import unittest


class RemoteCodeTestCase(unittest.TestCase):
    def test_remote_code(self):
        import os
        from chatbot import from_config
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
        from chatbot import from_bot_map_config, Converter
        from util import load_config
        config = load_config('config.json')
        cls.to_massages = Converter.to_messages
        cls.bot_map = from_bot_map_config(config['bot_map'])

    def test_gpt4_chat(self):
        print(self.bot_map['gpt-4'].chat(self.to_massages('你好')))
        print('=' * 16)
        for response in self.bot_map['gpt-4'].stream_chat(self.to_massages('你好')):
            print(response)

    def test_gpt3_chat(self):
        print(self.bot_map['gpt-3'].chat(self.to_massages('你好')))
        print('=' * 16)
        for response in self.bot_map['gpt-3'].stream_chat(self.to_massages('你好')):
            print(response)


class ConverterTestCase(unittest.TestCase):
    def test_convertor(self):
        from chatbot import Converter
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


if __name__ == '__main__':
    unittest.main()
