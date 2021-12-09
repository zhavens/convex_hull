"""Test script for finding tangents between two hulls."""

from ctypes import ArgumentError
import os
from datetime import datetime
from typing import List, Text

import matplotlib

from absl import app
from absl import flags
from absl import logging

from point import Point
import util
import convex_hull

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "left_hullfile", None, "The path to an input file containing x,y pairs of coordinates for the left hull.")


flags.DEFINE_string(
    "right_hullfile", None, "The path to an input file containing x,y pairs of coordinates for the left hull.")

flags.DEFINE_enum("tangent", "upper", [
                  "upper", "lower"], "Which of the tangents to find, \"upper\" or \"lower\".")

flags.DEFINE_bool(
    "show_plot", True, "Whether or not to plot the associated tangent points.")


def print_usage():
    """Prints usage information for the tool."""
    print("""
Usage: rightmost_point.py --left_hullfile=[hullfile] --right_hullfile=[hullfile]""")


def main(argv):
    del argv  # unused

    left_hull = util.fetch_input_points(FLAGS.left_hullfile)
    right_hull = util.fetch_input_points(FLAGS.right_hullfile)

    l_point = max(left_hull, key=lambda p: p.x)
    r_point = min(right_hull, key=lambda p: p.x)
    center_x = l_point.x + ((r_point.x - l_point.x) / 2)
    l_index = left_hull.index(l_point)
    r_index = right_hull.index(r_point)

    if FLAGS.tangent == "upper":
        l_index, r_index = convex_hull.find_upper_tangent(
            left_hull, l_index, right_hull, r_index, center_x)
    elif FLAGS.tangent == "lower":
        l_index, r_index = convex_hull.find_lower_tangent(
            left_hull, l_index, right_hull, r_index, center_x)
    else:
        raise ArgumentError()

    l_point = left_hull[l_index]
    r_point = right_hull[r_index]
    if (FLAGS.show_plot):
        util.show_plot(hulls=[left_hull, right_hull],
                       lines=[[l_point, r_point]],
                       labels={l_point: "l", r_point: "r"})
    logging.info(f"Left Point: {l_point} | Right Point: {r_point}")


if __name__ == "__main__":
    FLAGS.logtostderr = True
    flags.mark_flag_as_required('left_hullfile')
    flags.mark_flag_as_required('right_hullfile')

    matplotlib.use("TkAgg")
    app.run(main)
