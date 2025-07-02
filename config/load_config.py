import yaml

def load_config(path="configs/config.yaml"):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_config_win_env(path="configs/config.yaml"):
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config