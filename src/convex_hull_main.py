"""Main script for convex hull construction."""

import os
from datetime import datetime
from typing import List, Text

from absl import app
from absl import flags
from absl import logging

import util
import convex_hull

FLAGS = flags.FLAGS

flags.DEFINE_enum("algo", "gw", ["gw", "dc", "graham", "chan", "test"],
                  "The algorithm to generate the convex hull with.")

flags.DEFINE_string(
    "infile", None, "The path to an input file containing x,y pairs of coordinates.")

flags.DEFINE_string(
    "hull_outfile", None, "The output file path for writing.")

flags.DEFINE_string(
    "stats_outfile", None, "The path to a file to append runtime stats to."
)


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: convex_hull_main.py --infile=[infile] --outfile=[outfile] (--algo={gw, dc, graham, chan})""")


def main(argv):
    del argv  # unused

    if not FLAGS.infile:
        print("Error: --infile not specified.")
        print_usage()
        return 1

    points = util.fetch_input_points(FLAGS.infile)

    if logging.vlog_is_on(1):
        print(f"Input Points: {points}")

    start_time = datetime.now()

    hull = None
    if FLAGS.algo == "gw":
        hull = convex_hull.gift_wrapping(points)
    elif FLAGS.algo == "dc":
        hull = convex_hull.divide_and_conquer(points)
    elif FLAGS.algo == "grahams":
        hull = convex_hull.grahams_algorithm(points)
    elif FLAGS.algo == "chans":
        hull = convex_hull.chans_algorithm(points)
    elif FLAGS.algo == "test":
        hull = points
    else:
        raise NotImplementedError(
            f"--algo=\"{FLAGS.algo}\" not supported.")

    runtime = (datetime.now() - start_time).total_seconds()

    if not convex_hull.validate_hull(hull):
        print("Error: constructed hull is invalid.")
        return 1

    if FLAGS.hull_outfile:
        with open(FLAGS.hull_outfile, "w") as f:
            for p in hull:
                f.write(f"{p}\n")
    else:
        print(f"Hull Points: {hull}")

    if FLAGS.stats_outfile:
        with open(FLAGS.stats_outfile, "a") as f:
            f.write(f"{FLAGS.algo},{os.path.basename(FLAGS.infile)},{runtime}\n")
    return 0


if __name__ == "__main__":
    app.run(main)
