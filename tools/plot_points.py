"""A tool for quickly plotting points and hulls."""

import os
from datetime import datetime
from typing import List, Text

import matplotlib

from absl import app
from absl import flags
from absl import logging

import util

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "points_file", None, "The path to an input file containing x,y pairs of coordinates.")

flags.DEFINE_string(
    "hull_file", None, "The output file path for writing.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: plot_points.py [--points_file=[path]] [--hull_file=[path]])""")


def main(argv):
    del argv  # unused

    if (not FLAGS.points_file and not FLAGS.hull_file):
        logging.error(
            "At least one of --points_file or --hull_file must be specified.")

    points = util.fetch_input_points(
        FLAGS.points_file) if FLAGS.points_file else None
    hull = util.fetch_input_points(
        FLAGS.hull_file) if FLAGS.hull_file else None

    util.show_plot(points, hull)

    return 0


if __name__ == "__main__":
    FLAGS.logtostderr = True
    matplotlib.use("TkAgg")
    app.run(main)
