"""Main script for convex hull construction.

Sample Invocations ----------------------

Generates a hull using Chan's Algorithm, saves the hull to a file, and appends
runtime statistics to a CSV file:
$ convex_hull_main.py \
    --algo=chan \
    --infile=inputs/testpoints
    --hull_outfile=outputs/hull
    --stats_outfile=results/results.csv

Generates a hull using Graham's Scan and displays a plot of the results and
logs verbosely.
$ convex_hull_main.py \
    --algo=graham \
    --infile=inputs/testpoints
    --show_plot
    --verbose=1
"""

import cProfile
import pstats
import os
from datetime import datetime
from typing import List, Text

import matplotlib

from absl import app
from absl import flags
from absl import logging

import util
import convex_hull

FLAGS = flags.FLAGS

flags.DEFINE_enum("algo", "gw", ["gw", "dc", "grahams", "chans"],
                  "The algorithm to generate the convex hull with.")

flags.DEFINE_string(
    "infile", None, "The path to an input file containing x,y pairs of coordinates.")

flags.DEFINE_string(
    "hull_outfile", None, "The output file path for writing.")

flags.DEFINE_string(
    "stats_outfile", None, "The path to a file to append runtime stats to.")

flags.DEFINE_bool("profile_algo", False,
                  "Whether to profile the hull construction algorithm.")
flags.DEFINE_string("profile_outfile", None,
                    "The path to dump profiling information if enabled. If "
                    "unspecified, output will be dumped to stdout.")

flags.DEFINE_bool("show_plot", False,
                  "Whether to show the plot and hull after construction.")
flags.DEFINE_bool("validate_hull", True, "Whether to validate hull upon "
                  "completion.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: convex_hull_main.py --infile=[infile] --outfile=[outfile] (--algo={gw, dc, graham, chan})""")


def main(argv):
    del argv  # unused

    points = util.fetch_input_points(FLAGS.infile)
    logging.vlog(1, f"Input Points: {points}")

    algo = None
    if FLAGS.algo == "gw":
        algo = convex_hull.gift_wrapping
    elif FLAGS.algo == "dc":
        algo = convex_hull.divide_and_conquer
    elif FLAGS.algo == "grahams":
        algo = convex_hull.grahams_scan
    elif FLAGS.algo == "chans":
        algo = convex_hull.chans_algorithm
    else:
        raise NotImplementedError(
            f"--algo=\"{FLAGS.algo}\" not supported.")
    hull = None
    if FLAGS.profile_algo:
        with cProfile.Profile() as profiler:
            hull = algo(points)
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats()
        if FLAGS.profile_outfile:
            stats.dump_stats(FLAGS.profile_outfile)
    else:
        start_time = datetime.now()
        hull = algo(points)
        runtime = (datetime.now() - start_time).total_seconds()
        logging.info(f"Constructed the hull in {runtime}s.")

    if FLAGS.show_plot:
        util.show_plot(points, hulls=[hull])

    if FLAGS.hull_outfile:
        util.write_points(hull, FLAGS.hull_outfile)
    else:
        logging.info(f"Hull Points: {hull}")

    if FLAGS.validate_hull:
        if not convex_hull.validate_hull(hull, points):
            logging.error("Error: constructed hull is not convex.")
            return 1
        else:
            logging.info("Hull is valid!")

    if FLAGS.stats_outfile:
        logging.info(f"Writing run stats to {FLAGS.stats_outfile}")
        with open(FLAGS.stats_outfile, "a") as f:
            f.write(f"{FLAGS.algo},{os.path.basename(FLAGS.infile)},{runtime}\n")

    logging.info("Completed successfully!")
    return 0


if __name__ == "__main__":
    FLAGS.logtostderr = True
    flags.mark_flag_as_required('infile')

    matplotlib.use("TkAgg")
    app.run(main)
