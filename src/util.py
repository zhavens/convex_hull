"""A set of various util functions for the project."""

import os
from typing import Dict, List, Text

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


def show_plot(points: List[List[Point]] = [],
              hulls: List[List[Point]] = [],
              lines: List[List[Point]] = [],
              labels: Dict[int, Text] = {},
              label_hulls: bool = False,
              title: Text = None):
    """Shows a matplot lib plot for the given points and hull."""
    fig, ax = plt.subplots()

    if title:
        fig.canvas.set_window_title(title)
    if points:
        if isinstance(points[0], Point):
            points = [points]
        for pointset in points:
            ax.scatter([p.x for p in pointset], [p.y for p in pointset])

    if hulls:
        if isinstance(hulls[0], Point):
            hulls = [hulls]
        for hull in hulls:
            ax.scatter([p.x for p in hull], [p.y for p in hull])
            ax.plot([p.x for p in hull + [hull[0]]],
                    [p.y for p in hull + [hull[0]]])
            if label_hulls:
                for idx, p in enumerate(hull):
                    plt.annotate(idx, (p.x, p.y), ha='center')

    for l in lines:
        ax.plot([p.x for p in l], [p.y for p in l])

    for p, label in labels.items():
        plt.annotate(label, (p.x, p.y), ha='center')

    ax.grid(True)
    ax.margins(0.5, 0.5)
    fig.tight_layout()

    plt.axis("scaled")
    plt.show()


def show_jarvis_step(curr: Point, hull: List[Point], start: int, center: int, end: int):
    """Shows a matplot lib plot for the given step in a Jarvis search."""
    fig, ax = plt.subplots()

    ax.scatter([curr.x], [curr.y])
    ax.scatter([p.x for p in hull], [p.y for p in hull])
    ax.plot([p.x for p in hull + [hull[0]]],
            [p.y for p in hull + [hull[0]]])

    plt.annotate('S', (hull[start].x, hull[start].y), ha='center')
    plt.annotate('C', (hull[center].x, hull[center].y), ha='center')
    plt.annotate('E', (hull[end].x, hull[end].y), ha='center')

    ax.grid(True)
    ax.margins(0.5, 0.5)
    fig.tight_layout()

    plt.axis("scaled")
    plt.show()
