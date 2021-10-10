"""A set of various util functions for the project."""

import os
from typing import List, Text

import matplotlib.pyplot as plt

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
            components = sanitized.split(",")
            points.append(Point(int(components[0]), int(components[1])))
    return points


def show_plot(points: List[Point], hull: List[Point] = None):
    fig, ax = plt.subplots()
    ax.scatter([p.x for p in points], [p.y for p in points])

    if hull:
        ax.plot([p.x for p in hull + [hull[0]]],
                [p.y for p in hull + [hull[0]]])

    ax.grid(True)
    fig.tight_layout()

    plt.show()
