import json
import pathlib

__this_dir__ = pathlib.Path(__file__).parent


def get_package_meta():
    """Reads codemeta.json and returns it as dict"""
    with open(__this_dir__ / '../codemeta.json', 'r') as f:
        codemeta = json.loads(f.read())
    return codemeta
