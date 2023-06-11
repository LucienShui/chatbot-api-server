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


if __name__ == '__main__':
    unittest.main()
