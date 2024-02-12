from json import load


def load_config(config_file: str) -> dict:
    with open(config_file) as f:
        return load(f)
