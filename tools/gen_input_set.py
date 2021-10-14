"""Main script for convex hull construction.

Sample Invocations ----------------------

Generates a hull using Chan's distributionrithm, saves the hull to a file, and appends
runtime statistics to a CSV file:
$ convex_hull_main.py \
    --distribution=chan \
    --infile=inputs/testpoints
    --hull_outfile=outputs/hull
    --stats_outfile=results/results.csv

Generates a hull using Graham's Scan and displays a plot of the results and
logs verbosely.
$ convex_hull_main.py \
    --distribution=graham \
    --infile=inputs/testpoints
    --show_plot
    --verbose=1
"""

from datetime import datetime
import os
import random
from typing import List, Text

from absl import app
from absl import flags
from absl import logging

from point import Point
import util

FLAGS = flags.FLAGS

flags.DEFINE_enum("distribution", "random", ["random"],
                  "The distribution type for the generated points.")

flags.DEFINE_integer(
    "num_points", None, "The number of points to generate.")

flags.DEFINE_float(
    "max_coord", 100, "The absolute maxium value for a coordinate component.")

flags.DEFINE_string(
    "outfile", None, "The output file path for writing the points to.")

flags.DEFINE_bool("show_plot", False,
                  "Whether to show the points after generation.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: gen_input_set.py --distribution=[dist] --num_points=[num_points] --outfile=[infile]""")


def random_points(num_points: float, max_value: float):
    points = []
    for i in range(num_points):
        points.append(Point(random.uniform(0, max_value),
                      random.uniform(0, max_value)))
    return points


def main(argv):
    del argv  # unused

    points = None
    if FLAGS.distribution == "random":
        points = random_points(FLAGS.num_points, FLAGS.max_coord)
    else:
        raise NotImplementedError(
            f"--distribution=\"{FLAGS.distribution}\" not supported.")

    if FLAGS.show_plot:
        util.show_plot(points, None)

    util.write_points(points, FLAGS.outfile)


if __name__ == "__main__":
    FLAGS.logtostderr = True

    flags.register_validator('num_points',
                             lambda v: v > 0,
                             message='--num_points must be positive.')
    flags.mark_flag_as_required('outfile')

    app.run(main)
