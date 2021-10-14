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

flags.DEFINE_enum("distribution", "uniform", ["uniform", "normal"],
                  "The distribution type for the generated points.")

flags.DEFINE_integer(
    "num_points", None, "The number of points to generate.")

flags.DEFINE_float(
    "max_coord", 100, "The absolute maxium value for a coordinate component.")

flags.DEFINE_string(
    "outfile", None, "The output file path for writing the points to.")

flags.DEFINE_float("normal_stddev", None,
                   "The standarad deviation to use for a normal distribution. "
                   "Defaults to 1/10 of --max_coord.")

flags.DEFINE_bool("show_plot", False,
                  "Whether to show the points after generation.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: gen_input_set.py --distribution=[dist] --num_points=[num_points] --outfile=[infile]""")


def uniform_points(num_points: int, max_value: float):
    points = []
    for i in range(num_points):
        points.append(Point(random.uniform(0, max_value),
                            random.uniform(0, max_value)))
    return points


def normal_points(num_points: int, mean: float, stddev: float):
    points = []
    for i in range(num_points):
        points.append(Point(random.gauss(mean, stddev),
                            random.gauss(mean, stddev)))
    return points


def main(argv):
    del argv  # unused

    points = None
    if FLAGS.distribution == "uniform":
        points = uniform_points(FLAGS.num_points, FLAGS.max_coord)
    elif FLAGS.distribution == "normal":
        stddev = FLAGS.normal_stddev if FLAGS.normal_stddev else FLAGS.max_coord/10
        points = normal_points(FLAGS.num_points, FLAGS.max_coord/2, stddev)
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
