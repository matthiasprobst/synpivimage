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

    @classmethod
    @abc.abstractmethod
    def load_jsonld(cls, filename: Union[str, pathlib.Path]):
        """Load the component from JSON-LD"""


def load_jsonld(cls, iri, filename) -> List[Component]:
    """Load the component from JSON-LD"""
    filename = pathlib.Path(filename)
    import ontolutils
    data = ontolutils.dquery(
        iri,
        filename,
        context={'pivmeta': 'https://matthiasprobst.github.io/pivmeta#'}
    )
    if data is None:
        raise ValueError(f"Could not load camera from {filename}.")

    components = []
    for d in data:
        components.append(
            cls(**{p['label']: p.get('hasNumericalValue', p.get('hasStringValue', None)) for p in
                   d['hasParameter']})
        )

    return components
