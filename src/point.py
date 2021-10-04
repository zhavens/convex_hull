from dataclasses import dataclass

from typing import List, Text


@dataclass
class Point:
    """A simple point struct for holding coordinates."""
    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"
