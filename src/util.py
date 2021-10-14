"""A set of various util functions for the project."""

import os
from typing import List, Text

import matplotlib.pyplot as plt

from point import Point


def fetch_input_points(path: Text) -> List[Point]:
    """Fetches input points from the given file.

    Points should be represented as "x,y" pairs, one per line. Any parentheses
    will be stripped.

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
            points.append(Point(float(components[0]), float(components[1])))
    return points


def write_points(points: List[Point], path: Text):
    """Write the points to the given file."""
    with open(path, "w") as f:
        for p in points:
            f.write(f"{p}\n")


def show_plot(points: List[Point], hull: List[Point] = None):
    """Shows a matplot lib plot for the given points and hull."""
    fig, ax = plt.subplots()
    if points:
        ax.scatter([p.x for p in points], [p.y for p in points])

    if hull:
        ax.scatter([p.x for p in hull], [p.y for p in hull])
        ax.plot([p.x for p in hull + [hull[0]]],
                [p.y for p in hull + [hull[0]]])
        for idx, p in enumerate(hull):
            plt.annotate(idx, (p.x, p.y), ha='center')

    ax.grid(True)
    ax.margins(0.5, 0.5)
    fig.tight_layout()

    plt.axis("scaled")
    plt.show()
