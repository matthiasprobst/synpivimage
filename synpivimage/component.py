import abc
import json
import pathlib


class Component(abc.ABC):

    def save(self, filename: str):
        filename = pathlib.Path(filename).with_suffix('.json')
        with open(filename, 'w') as f:
            json.dump(self.model_dump(), f, indent=4)
