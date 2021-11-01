"""Main script for convex hull validation."""

from absl import app
from absl import flags
from absl import logging

import util
import convex_hull

FLAGS = flags.FLAGS

flags.DEFINE_string("hull_file", None, "The path to an input file containing x,y "
                    "pairs of coordinates defining a convex hull.")
flags.DEFINE_string("points_file", None, "The path to an input file containing x,y "
                    "pairs of coordinates defining a set of points hull.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: validate_hull.py --infile=[infile]""")


def main(argv):
    del argv  # unused

    hull = util.fetch_input_points(FLAGS.hull_file)
    points = util.fetch_input_points(
        FLAGS.points_file) if FLAGS.points_file else []

    if logging.vlog_is_on(1):
        print(f"Hull Points: {hull}")

    if convex_hull.validate_hull(hull, points):
        print("Hull is valid.")
        return 0
    else:
        print("Hull is invalid.")
        return 1


if __name__ == "__main__":
    flags.mark_flag_as_required('hull_file')

    app.run(main)
