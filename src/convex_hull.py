
from typing import List

from absl import logging
import numpy as np

from point import Point


def gift_wrapping(points: List[Point]) -> List[Point]:
    """Implementation of the gift-wrapping algorithm."""
    raise NotImplementedError()


def divide_and_conquer(points: List[Point]) -> List[Point]:
    """Implementation of the divide-and-conquer algorithm."""
    raise NotImplementedError()


def grahams_algorithm(points: List[Point]) -> List[Point]:
    """Implementation of Graham's algorithm."""
    raise NotImplementedError()


def chans_algorithm(points: List[Point]) -> List[Point]:
    """Implementation of Chan's algorithm."""
    raise NotImplementedError()


def is_convex(points: List[Point]) -> bool:
    if not points:
        return False
    if len(points) < 3:
        return True

    logging.vlog(1, "Testing convexity...")

    winding = None
    for i in range(0, len(points)):
        a = points[i]
        b = points[(i+1) % len(points)]
        c = points[(i+2) % len(points)]
        cp = np.cross(b-a, c-b)

        logging.vlog(1, f"{b}x{c} = {cp}")
        if cp == 0:
            continue
        elif not winding:
            winding = cp
        elif (cp > 0) != (winding > 0):
            return False
    return True


def validate_hull(points: List[Point]) -> bool:
    """Validates the convex hull provded as a series of points."""
    if not points:
        return False
    if len(points) < 3:
        return True

    return is_convex(points)
