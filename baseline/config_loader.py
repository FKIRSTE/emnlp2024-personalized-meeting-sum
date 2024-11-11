import json

def load_config(config_path):
    """
    Load a JSON configuration file from the given path.
    :param config_path: The path to the JSON configuration file.
    :return: The configuration as a dictionary.
    """
    with open(config_path, encoding='utf-8') as config_file:
        config = json.load(config_file)
    return config
