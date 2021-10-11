"""Main script for convex hull validation."""

from absl import app
from absl import flags
from absl import logging

import util
import convex_hull

FLAGS = flags.FLAGS

flags.DEFINE_string("infile", None, "The path to an input file containing x,y "
                    "pairs of coordinates.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: validate_hull.py --infile=[infile]""")


def main(argv):
    del argv  # unused

    if not FLAGS.infile:
        print("Error: --infile not specified.")
        print_usage()
        return 1

    hull = util.fetch_input_points(FLAGS.infile)

    if logging.vlog_is_on(1):
        print(f"Input Points: {hull}")

    if convex_hull.validate_hull(hull):
        print("Hull is valid.")
        return 0
    else:
        print("Hull is invalid.")
        return 1


if __name__ == "__main__":
    app.run(main)
