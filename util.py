from json import load
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger: logging.Logger = logging.getLogger('app')


def load_config(config_file: str) -> dict:
    with open(config_file) as f:
        return load(f)
