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
    "hull_dir", None, "The output dir to dump the hull path into.")

flags.DEFINE_string(
    "stats_outfile", None, "The path to a file to append runtime stats to.")

flags.DEFINE_bool("profile_algo", False,
                  "Whether to profile the hull construction algorithm.")
flags.DEFINE_string("profile_dir", None,
                    "The dir to dump profiling information in if enabled. If "
                    "unspecified, output will be dumped to stdout.")

flags.DEFINE_bool("show_plot", False,
                  "Whether to show the plot and hull after construction.")
flags.DEFINE_bool("validate_hull", False, "Whether to validate hull upon "
                  "completion.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: convex_hull_main.py --infile=[infile] --outfile=[outfile] (--algo={gw, dc, graham, chan})""")


def main(argv):
    del argv  # unused

    input_name = os.path.basename(FLAGS.infile)
    points = util.fetch_input_points(FLAGS.infile)
    logging.vlog(4, f"Input Points: {points}")

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
        if FLAGS.profile_dir:
            profile_path = os.path.join(FLAGS.profile_dir,
                                        f"{FLAGS.algo}_{input_name}")
            logging.info(f"Dumping profiling stats to {profile_path}.")
            stats.dump_stats(profile_path)
    else:
        start_time = datetime.now()
        hull = algo(points)
        runtime = (datetime.now() - start_time).total_seconds()
        logging.info(f"Constructed the hull in {runtime}s.")
        if FLAGS.stats_outfile:
            logging.info(f"Writing run stats to {FLAGS.stats_outfile}")
            with open(FLAGS.stats_outfile, "a") as f:
                f.write(f"{FLAGS.algo},{input_name},{runtime}\n")

    if FLAGS.hull_dir:
        hull_path = os.path.join(FLAGS.hull_dir, f"{FLAGS.algo}_{input_name}")
        logging.info(f"Writing output hull to {hull_path}.")
        util.write_points(hull, hull_path)
    else:
        logging.info(f"Hull Points: {hull}")

    if FLAGS.show_plot:
        logging.info("Showing plot for hull. Close plot to continue...")
        util.show_plot(points, hulls=[hull])

    if FLAGS.validate_hull:
        if not convex_hull.validate_hull(hull, points):
            logging.error("Error: constructed hull is not convex.")
            return 1
        else:
            logging.info("Hull is valid!")

    logging.info("Completed successfully!")
    return 0


if __name__ == "__main__":
    FLAGS.logtostderr = True
    flags.mark_flag_as_required('infile')

    # May be necessary on certain systems to specify the MPL backend.
    # See https://matplotlib.org/stable/users/explain/backends.html for details.
    # matplotlib.use("TkAgg")
    app.run(main)
