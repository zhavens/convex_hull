from dataclasses import dataclass
from typing import List, Text

import numpy as np


@dataclass
class Point:
    """A simple point struct for holding coordinates."""
    x: float
    y: float

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __getitem__(self, key) -> float:
        return [self.x, self.y][key]

    def __array__(self, dtype=None):
        return np.array([self.x, self.y], dtype=float)
