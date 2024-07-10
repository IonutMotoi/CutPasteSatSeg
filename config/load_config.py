import yaml


# Function to load yaml configuration file
def load_config(config_path="./config/config.yaml"):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def print_config(config):
    print("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("")