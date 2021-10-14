"""A library for creating and validating convex hulls."""

from operator import attrgetter
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


def grahams_scan(points: List[Point]) -> List[Point]:
    """Implementation of Graham's scan."""

    hull = []

    def _IsCCW(c, a, b):
        return np.cross(a-c, b-c) >= 0

    p0 = min(points, key=lambda p: (p.y, p.x))
    points = sorted(points, key=lambda p: np.dot(
        p-p0, [1, 0])/np.linalg.norm(p-p0) if p != p0 else 1, reverse=True)
    # dots = [np.dot(p-p0, [1, 0])/np.linalg.norm(p-p0)
    #         for p in points if p != p0]

    logging.vlog(1, f"p0: {p0}")
    logging.vlog(1, f"Sorted points: {points}")

    for p in points:
        while len(hull) > 1 and not _IsCCW(hull[-2], hull[-1], p):
            hull.pop()
        hull.append(p)
        logging.vlog(2, hull)

    return hull


def chans_algorithm(points: List[Point]) -> List[Point]:
    """Implementation of Chan's algorithm."""
    raise NotImplementedError()


def is_convex(points: List[Point]) -> bool:
    """Determines whether the set of verices forms a convex poly.

    Uses the cross product to determine the angle between adjacent edges. If
    all angles have a consistent winding (left- or right-handed), then the poly
    is convex. A degenerate poly is not considered convex.

    https://en.wikipedia.org/wiki/Cross_product#Computational_geometry

    Args:
        points: The list of vertices that form the polygon.
    Returns:
        Whether the poly is convex.
    """
    if not points or len(points) < 3:
        return False

    winding = None
    for i in range(0, len(points)):
        a = points[i]
        b = points[(i+1) % len(points)]
        c = points[(i+2) % len(points)]
        cp = np.cross(b-a, c-b)

        if cp == 0:
            continue
        elif not winding:
            winding = cp
        elif (cp > 0) != (winding > 0):
            return False
    return True


def point_in_poly(p1: Point, poly: List[Point]) -> bool:
    """Determines whether the point is in the given polygon.

    Uses a ray-casting method, determining the number of interstions the ray
    has with edges of the poly to indicate whether the point falls within it.
    Intersection testing uses a stripped version of the Bezier-based
    segment-segment equation, as the ray is drawn horizonally, so all
    (p1.y - p2.y) terms will be zero and can be removed.

    https://en.wikipedia.org/wiki/Line_line_intersection#Given_two_points_on_each_line_segment

    Args:
        p1: The point to query.
        poly: An ordered list of vertices that make up the poly.
    Returns:
        Whether the point is within the poly.
    """
    if len(poly) < 3:
        return False

    p2 = Point(MAX_COORD, p1.y)
    num_intersections = 0
    for i in range(len(poly)):
        q1 = poly[i]
        q2 = poly[(i+1) % len(poly)]

        denominator = (p1.x - p2.x)*(q1.y - q2.y)
        if denominator == 0:
            if p1.y == q1.y and q1.x <= p1.x <= q2.x:
                # Point lies _on_ a line of the poly.
                return True
            else:
                continue

        t = float((p1.x - q1.x)*(q1.y - q2.y) -
                  (p1.y - q1.y)*(q1.x - q2.x)) / denominator
        u = float(-(p1.y - q1.y)*(p1.x - p2.x)) / denominator

        if (0 <= t <= 1) and (0 <= u <= 1):
            logging.vlog(2, f'Intersecton: {p1} -> {q1}-{q2}')
            num_intersections += 1
    return num_intersections % 2 == 1


def validate_hull(points: List[Point], hull: List[Point]) -> bool:
    """Validates the convex hull provded as a series of points.

    Checks both that the hull is convex and that all points are either a part of
    the hull or within its confines.

    Args:
        points: The full point set the hull was created for
        hull: The list of ordered points that make up the convex hull.
    Returns:
        Whether the hull is valid for the given points.
    """
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
