import abc
import enum
import json
import pathlib
from typing import List, Union


class SaveFormat(enum.Enum):
    """Save format enumerator"""
    JSON = 'json'
    JSONLD = 'json-ld'


class Component(abc.ABC):
    """Abstract class for a component, e.g. a laser or a camera"""

    # def save(self, filename: Union[str, pathlib.Path],
    #          format: SaveFormat = SaveFormat.JSON) -> pathlib.Path:
    #     """Save the component to a format specified by `format`"""
    #     filename = pathlib.Path(filename)
    #     if SaveFormat(format) == SaveFormat.JSON:
    #         return self.save_json(filename)
    #     if SaveFormat(format) == SaveFormat.JSONLD:
    #         return self.save_jsonld(filename)

    def save_json(self, filename: Union[str, pathlib.Path]):
        """Save the component to JSON"""
        filename = pathlib.Path(filename).with_suffix('.json')  # .with_suffix('.json')
        with open(filename, 'w') as f:
            json.dump(self.model_dump(), f, indent=4)
        return filename

    @abc.abstractmethod
    def save_jsonld(self, filename: Union[str, pathlib.Path]) -> pathlib.Path:
        """Save the component to JSON"""

    # @classmethod
    # def load(cls, filename: Union[str, pathlib.Path]):
    #     """Load the component from JSON"""
    #     filename = pathlib.Path(filename).with_suffix('.json')
    #     with open(filename) as f:
    #         data = json.load(f)
    #     return cls(**data)


# def save_multiple(components: List[Component],
#                   filename: Union[str, pathlib.Path]):
#     """Saves multiple components into a single file"""
#     filename = pathlib.Path(filename).with_suffix('.json')
#     with open(filename, 'w') as f:
#         for component in components:
#             json.dump(component.model_dump(), f, indent=4)
