
from re import I
from typing import List
import sys

from absl import logging
import numpy as np

from point import Point

# The maximum coordinate value we want to use in our plane.
MAX_COORD = 1e9


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
    if not points or len(points) < 3:
        return False

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


def point_in_poly(p1: Point, poly: List[Point]) -> bool:
    """Determines whether the point is in the given polygon."""
    if len(poly) < 3:
        return False

    p2 = Point(MAX_COORD, p1.y)
    num_intersections = 0
    for i in range(len(poly)):
        q1 = poly[i]
        q2 = poly[(i+1) % len(poly)]

        den = (p1.x - p2.x)*(q1.y - q2.y)
        if den == 0:
            if p1.y == q1.y and q1.x <= p1.x <= q2.x:
                # Point lies _on_ a line of the poly.
                return True
            else:
                continue

        t = float((p1.x - q1.x)*(q1.y - q2.y) -
                  (p1.y - q1.y)*(q1.x - q2.x)) / den
        u = float(-(p1.y - q1.y)*(p1.x - p2.x)) / den

        if (0 <= t <= 1) and (0 <= u <= 1):
            logging.info(f'Intersecton: {p1} -> {q1}-{q2}')
            num_intersections += 1
    return num_intersections % 2 == 1


def validate_hull(points: List[Point], hull: List[Point]) -> bool:
    """Validates the convex hull provded as a series of points."""
    if not points:
        return False
    if len(points) < 3:
        return True

    if not is_convex(hull):
        return False

    for p in points:
        if not point_in_poly(p, hull):
            return False

    return True
