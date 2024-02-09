import abc
import json
import pathlib
from typing import List, Union


class Component(abc.ABC):
    """Abstract class for a component, e.g. a laser or a camera"""

    def save(self, filename: Union[str, pathlib.Path]) -> pathlib.Path:
        """Save the component to JSON"""
        filename = pathlib.Path(filename).with_suffix('.json')
        with open(filename, 'w') as f:
            json.dump(self.model_dump(), f, indent=4)
        return filename


def save_multiple(components: List[Component],
                  filename: Union[str, pathlib.Path]):
    """Saves multiple components into a single file"""
    filename = pathlib.Path(filename).with_suffix('.json')
    with open(filename, 'w') as f:
        for component in components:
            json.dump(component.model_dump(), f, indent=4)
