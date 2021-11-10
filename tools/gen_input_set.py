"""Main script for generating input sets."""

from datetime import datetime
import math
import os
import random
from typing import List, Text

from absl import app
from absl import flags
from absl import logging

from point import Point
import util

FLAGS = flags.FLAGS

flags.DEFINE_enum("distribution", "uniform", ["uniform", "normal", "clustered"],
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

flags.DEFINE_integer("clustered_num_clusters", 1, "The number of clusters to "
                     "generate points around.")
flags.DEFINE_float("clustered_max_dist_from_center", 10, "How far points can "
                   "be from the center of a cluster.")

flags.DEFINE_bool("show_plot", False,
                  "Whether to show the points after generation.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: gen_input_set.py --distribution=[dist] --num_points=[num_points] --outfile=[infile]""")


def uniform_points(num_points: int, max_coord: float):
    points = []
    for i in range(num_points):
        points.append(Point(random.uniform(0, max_coord),
                            random.uniform(0, max_coord)))
    return points


def normal_points(num_points: int, mean: float, stddev: float):
    points = []
    for i in range(num_points):
        points.append(Point(random.gauss(mean, stddev),
                            random.gauss(mean, stddev)))
    return points


def clustered_points(num_points: int, num_clusters: int, max_coord: float,
                     max_from_center: float):
    points = []
    centers = [Point(random.uniform(0, max_coord),
                     random.uniform(0, max_coord)) for i in range(num_clusters)]

    for i in range(num_points):
        center = int(random.uniform(0, num_clusters))
        dist = random.uniform(0, max_from_center)
        angle = random.uniform(0, math.pi*2)
        points.append(Point(centers[center].x + (math.cos(angle)*dist),
                            centers[center].y + (math.sin(angle)*dist)))

    return points


def main(argv):
    del argv  # unused

    points = None
    if FLAGS.distribution == "uniform":
        points = uniform_points(FLAGS.num_points, FLAGS.max_coord)
    elif FLAGS.distribution == "normal":
        stddev = FLAGS.normal_stddev if FLAGS.normal_stddev else FLAGS.max_coord/10
        points = normal_points(FLAGS.num_points, FLAGS.max_coord/2, stddev)
    elif FLAGS.distribution == "clustered":
        points = clustered_points(FLAGS.num_points,
                                  FLAGS.clustered_num_clusters,
                                  FLAGS.max_coord,
                                  FLAGS.clustered_max_dist_from_center)
    else:
        raise NotImplementedError(
            f"--distribution=\"{FLAGS.distribution}\" not supported.")

    if FLAGS.show_plot:
        util.show_plot(points, None)

    if FLAGS.outfile:
        util.write_points(points, FLAGS.outfile)
        logging.info(f"Wrote points to {FLAGS.outfile}")


if __name__ == "__main__":
    FLAGS.logtostderr = True

    flags.register_validator('num_points',
                             lambda v: v > 0,
                             message='--num_points must be positive.')
    # flags.mark_flag_as_required('outfile')

    app.run(main)
