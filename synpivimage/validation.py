from annotated_types import Gt
from dataclasses import dataclass
from typing_extensions import Annotated

PositiveInt = Annotated[int, Gt(0)]
PositiveFloat = Annotated[float, Gt(0)]


@dataclass
class ValueRange:
    lo: int
    hi: int
