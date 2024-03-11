import yaml


def get_config() -> dict:
    with open("config.yaml", "r") as file:
        data = yaml.safe_load(file)
    return data
