import logging
import os
import json
from datetime import datetime
import fcntl


class CustomFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            'message': record.msg if isinstance(record.msg, dict) else record.getMessage(),
            'level': record.levelname,
            'create_time': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            'name': record.name
        }
        str_log = json.dumps(log, ensure_ascii=False, separators=(',', ':'))
        return str_log


class CustomStreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.formatter = CustomFormatter()


class CustomFileHandler(logging.Handler):
    terminator = '\n'

    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir: str = log_dir
        self.formatter = CustomFormatter()

    def emit(self, record: logging.LogRecord) -> None:
        str_log = self.formatter.format(record)
        date = datetime.fromtimestamp(record.created)
        filename = date.strftime('%Y%m%d.log')
        with open(os.path.join(self.log_dir, filename), 'a', encoding='utf-8') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(str_log + self.terminator)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


LOG_DIR = 'logs'

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        CustomFileHandler(LOG_DIR),
        CustomStreamHandler()
    ]
)
logger: logging.Logger = logging.getLogger('app')
