import unittest


class RemoteCodeTestCase(unittest.TestCase):
    def test_remote_code(self):
        import os
        from chatbot import import_remote
        base_dir = '/tmp/7Bfa7T9gH3nwBi5a'
        os.system(f'rm -rf {base_dir}')
        os.system(f'mkdir -p {base_dir}')
        with open(os.path.join(base_dir, 'model.py'), 'w') as f:
            f.write("""
from chatbot import ChatBotBase
class Bot(ChatBotBase):
    def __init__(self):
        super().__init__()
    def chat(self, query: str, history: list = None, system: str = None, parameters: dict = None) -> str:
        return query""")
        path = os.path.expanduser(os.path.join(base_dir, 'model.Bot'))
        obj = import_remote(path, {})
        text = 'Hello World!'
        os.system(f'rm -rf {base_dir}')
        self.assertEqual(text, obj.chat(text))


if __name__ == '__main__':
    unittest.main()
