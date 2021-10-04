import os
from typing import List, Text

from point import Point


def fetch_input_points(path: Text) -> List[Point]:
    """Fetches input points from the given file.

    Points should be represented as "x,y" pairs, one per line. Any parentheses will be stripped.

    Args:
      path: The path to the input file
    Returns:
      A list of Point structs.
    """
    points = []

    fullpath = path if os.path.isabs(path) else os.path.join(os.curdir, path)

    with open(fullpath, "r") as f:
        for l in f:
            sanitized = l.strip("(").strip(")\n")
            points.append(Point(*sanitized.split(",")))
    return points
