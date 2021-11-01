"""Main script for convex hull construction.

Sample Invocations ----------------------

Generates a hull using Chan's Algorithm, saves the hull to a file, and appends
runtime statistics to a CSV file:
$ convex_hull_main.py \
    --algo=chan \
    --hullfile=inputs/testhull
    --hull_outfile=outputs/hull
    --stats_outfile=results/results.csv

Generates a hull using Graham's Scan and displays a plot of the results and
logs verbosely.
$ convex_hull_main.py \
    --algo=graham \
    --hullfile=inputs/testhull
    --show_plot
    --verbose=1
"""

import os
from datetime import datetime
from typing import List, Text

import matplotlib

from absl import app
from absl import flags
from absl import logging

from point import Point
import util
import convex_hull

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "hullfile", None, "The path to an input file containing x,y pairs of coordinates.")

flags.DEFINE_list(
    "point", None, "The coordinates of the reference point.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: rightmost_point.py --hullfile=[hullfile] --point=[outfile]""")


def main(argv):
    del argv  # unused

    p = Point(float(FLAGS.point[0].strip("(")),
              float(FLAGS.point[1].strip(")")))
    hull = util.fetch_input_points(FLAGS.hullfile)
    rightmost = convex_hull.find_rightmost_in_hull(p, hull)
    logging.info(f"Rightmost point: {rightmost}")


if __name__ == "__main__":
    FLAGS.logtostderr = True
    flags.mark_flag_as_required('hullfile')
    flags.register_validator('point',
                             lambda l: len(l) == 2,
                             message='--point must be "x,y" coordinate pair.')

    matplotlib.use("TkAgg")
    app.run(main)
