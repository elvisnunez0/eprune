from typing import Dict

import yaml


def flatten_dict(d: Dict, prefix: str = "", separator: str = ".") -> Dict:
    """
    Flattens the provided dictionary so that if a key has a dictionary as
    a value, the keys will be merged so that we construct a single dictionary.
    Keys are separated with @separator.

    For exmple,
        {'common': {'task': 'classification', 'logger': {'frequency': 200}}
        becomes:
        {'common.task': classification, 'common.logger.frequency': 200}

    Args:
        d: The dictionary to flatten.
        prefix: The current key prefix. Should be '' when first calling this function,
            redefined recursively.
        separator: The separator character between keys.

    Returns:
        flattened_dict: The flattened dictionary.
    """
    flattened_dict = {}
    for key, value in d.items():
        if prefix == "":
            current_prefix = prefix + key
        else:
            current_prefix = prefix + separator + key

        if isinstance(value, Dict):
            flattened_dict.update(flatten_dict(value, prefix=current_prefix))
        else:
            flattened_dict[current_prefix] = value

    return flattened_dict


def load_config(cfg_path: str, flatten: bool = False) -> Dict:
    """
    Loads a yaml file and optionally flattens it into a single dictionary
    (with no subdictionaries).

    Args:
        config_path: The path to the yaml config file.
        flatten: Whether the dictionary of the loaded config file should be flattened.
    """
    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)

    if flatten:
        cfg = flatten_dict(cfg)

    return cfg
