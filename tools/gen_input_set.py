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

MAX_EPSILON = 1e-12

flags.DEFINE_enum("distribution", "uniform",
                  ["uniform", "normal", "clustered", "circle"],
                  "The distribution type for the generated points.")

flags.DEFINE_integer(
    "num_points", None, "The number of points to generate.")

flags.DEFINE_float(
    "max_coord", 100, "The absolute maxium value for a coordinate component.")

flags.DEFINE_string(
    "outfile", None, "The output file path for writing the points to.")

flags.DEFINE_string("outdir", None, "If specified and --outfile is empty, dir "
                    "to place point files in with generated names.")

flags.DEFINE_float("normal_stddev", None,
                   "The standarad deviation to use for a normal distribution. "
                   "Defaults to 1/10 of --max_coord.")

flags.DEFINE_integer("clustered_num_clusters", 1, "The number of clusters to "
                     "generate points around.")
flags.DEFINE_float("clustered_relative_size", 2.5, "The relative size of the "
                   "cluster based on number of clusters and max coord value.")

flags.DEFINE_boolean("circle_general_position", True, "Whether to ensure that "
                     "circle points are in general position by applying some "
                     "small epsilon perturbation.")

flags.DEFINE_bool("randomize_order", False, "Whether the points should be "
                  "randomly reordered after generation.")

flags.DEFINE_bool("show_plot", False,
                  "Whether to show the points after generation.")

flags.DEFINE_bool("add_bounding_box", False, "Whether to add a bounding box "
                  "around the points generated by the given distribution.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: gen_input_set.py --distribution=[dist] --num_points=[num_points] --outfile=[infile]""")


def uniform_points(num_points: int, max_coord: float) -> List[Point]:
    points = []
    for i in range(num_points):
        points.append(Point(random.uniform(0, max_coord),
                            random.uniform(0, max_coord)))
    return points


def normal_points(num_points: int, mean: float, stddev: float) -> List[Point]:
    points = []
    for i in range(num_points):
        points.append(Point(random.gauss(mean, stddev),
                            random.gauss(mean, stddev)))
    return points


def clustered_points(num_points: int, num_clusters: int, max_coord: float,
                     relative_size: float) -> List[Point]:
    points = []
    centers = [Point(random.uniform(0, max_coord),
                     random.uniform(0, max_coord)) for i in range(num_clusters)]

    max_from_center = max_coord * relative_size / num_clusters

    logging.info(f"CENTERMAX: {max_from_center}")

    for i in range(num_points):
        center = int(random.uniform(0, num_clusters))
        dist = random.uniform(0, max_from_center)
        angle = random.uniform(0, math.pi*2)
        points.append(Point(centers[center].x + (math.cos(angle)*dist),
                            centers[center].y + (math.sin(angle)*dist)))

    return points


def circle_points(num_points: int, max_coord: float) -> List[Point]:
    points = []
    for i in range(num_points):
        angle = (i/num_points) * 2 * math.pi
        x = max_coord * math.cos(angle)
        y = max_coord * math.sin(angle)
        if FLAGS.circle_general_position:
            x = x + random.uniform(0, MAX_EPSILON)
            y = y + random.uniform(0, MAX_EPSILON)
        points.append(Point(x, y))
    return points


def _si_prefix(num: int) -> Text:
    exp = math.log10(num)
    if exp < 3:
        return f"{num}"
    elif exp < 6:
        return f"{int(num/1000)}K"
    elif exp < 9:
        return f"{int(num/1000)}M"
    else:
        return f"{num}"


def main(argv):
    del argv  # unused

    points = None
    filename = None
    if FLAGS.distribution == "uniform":
        points = uniform_points(FLAGS.num_points, FLAGS.max_coord)
        filename = f"uniform{_si_prefix(FLAGS.num_points)}"
    elif FLAGS.distribution == "normal":
        stddev = FLAGS.normal_stddev if FLAGS.normal_stddev else FLAGS.max_coord/10
        points = normal_points(FLAGS.num_points, FLAGS.max_coord/2, stddev)
        filename = f"normal{_si_prefix(FLAGS.num_points)}"
    elif FLAGS.distribution == "clustered":
        points = clustered_points(FLAGS.num_points,
                                  FLAGS.clustered_num_clusters,
                                  FLAGS.max_coord,
                                  FLAGS.clustered_relative_size)
        filename = f"clustered{_si_prefix(FLAGS.num_points)}_{FLAGS.clustered_num_clusters}c"
    elif FLAGS.distribution == "circle":
        points = circle_points(FLAGS.num_points, FLAGS.max_coord)
        filename = f"circle{_si_prefix(FLAGS.num_points)}"
    else:
        raise NotImplementedError(
            f"--distribution=\"{FLAGS.distribution}\" not supported.")

    if FLAGS.add_bounding_box:
        min_x = min([p.x for p in points]) - 0.01
        max_x = max([p.x for p in points]) + 0.01
        min_y = min([p.y for p in points]) - 0.01
        max_y = max([p.y for p in points]) + 0.01
        points.append(Point(min_x, min_y))
        points.append(Point(min_x, max_y))
        points.append(Point(max_x, min_y))
        points.append(Point(max_x, max_y))
        filename += "_box"

    if FLAGS.randomize_order:
        random.shuffle(points)
        filename += "_rdm"

    if FLAGS.show_plot:
        util.show_plot(points, None)

    if FLAGS.outfile:
        util.write_points(points, FLAGS.outfile)
        logging.info(f"Wrote points to {FLAGS.outfile}")
    elif FLAGS.outdir:
        path = os.path.join(FLAGS.outdir, filename)
        util.write_points(points, path)
        logging.info(f"Wrote points to {path}")
    else:
        logging.warning(f"No paths specified, points not being written.")


if __name__ == "__main__":
    FLAGS.logtostderr = True

    flags.register_validator('num_points',
                             lambda v: v > 0,
                             message='--num_points must be positive.')

    app.run(main)
