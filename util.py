from json import load
import logging
from logging.handlers import TimedRotatingFileHandler
import os

LOG_DIR = 'logs'

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

file_handler = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, 'app.log'), when='MIDNIGHT', interval=1, backupCount=0, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        file_handler,
        logging.StreamHandler()
    ]
)
logger: logging.Logger = logging.getLogger('app')


def load_config(config_file: str) -> dict:
    with open(config_file) as f:
        return load(f)
