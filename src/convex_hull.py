"""A library for creating and validating convex hulls."""

import math
from operator import attrgetter
from typing import List, Tuple
import sys

from absl import flags
from absl import logging
import numpy as np

from point import Point
import util

FLAGS = flags.FLAGS

flags.DEFINE_bool("plot_errors", False, "Whether to show plots when there "
                  "are validation or construction errors.")

flags.DEFINE_bool("verbose_plotting", False, "Whether to plot when logging "
                  "verbosely.")

flags.DEFINE_bool("grahams_split_hull", False, "Whether Graham's scan should "
                  "construct the hull from upper and lower pieces. Causes "
                  "point sorting by x-coordinate instead of radial angle.")

flags.DEFINE_bool("chans_eliminate_points", False, "Whether to eliminate "
                  "points from the input set if they aren't part of any "
                  "calculated hulls.")
flags.DEFINE_bool("chans_merge_hulls", False, "Whether to merge previously "
                  "calculated hulls when increasing subset size.")
flags.DEFINE_integer("chans_initial_t", 1, "The initial |t| parameter to use "
                     "when estimating |m| using the squaring scheme.")

# The maximum coordinate value we want to use in our plane.
MAX_COORD = 100

COLINEAR = 0
RIGHT_TURN = -1
LEFT_TURN = 1
CCW = 1
CW = -1


def vplot_is_on(level: int):
    return logging.vlog_is_on(level) and FLAGS.verbose_plotting


def find_orientation(curr: Point, prev: Point, next: Point) -> int:
    """Finds the orientation of the angle within the points.

    Args:
        curr: The apex or root point of the angle.
        prev: The point before the root
        next: The point after the root
    Returns:
        The orientation between the two vectors.
    """
    val = (prev.x-curr.x)*(next.y-curr.y) - (prev.y-curr.y)*(next.x-curr.x)
    if (val == 0):
        return COLINEAR
    elif val > 0:
        return LEFT_TURN
    else:
        return RIGHT_TURN


def angle_between(a: Point, b: Point, c: Point):
    """Finds the angle between the vectors defined by `ba` and `bc`.

    Args:
        a: The first point.
        b: The second (apex) point.
        c: The final point.
    Returns:
        The angle between the two vectors in radians. 
    """
    v1 = a - b
    v2 = c - b
    return np.arccos(np.dot(v1, v2) /
                     (np.linalg.norm(v1) * np.linalg.norm(v2)))


def find_rightmost_in_hull(p: Point, hull: List[Point]) -> Point:
    """Finds the rightmost point in a given hull from p.

    Uses a modified Jarvis scan to be able to binary search the hull, finding
    the rightmost point in O(logh) time.

    Adapted from https://iq.opengenus.org/chans-algorithm-convex-hull/

    Args:
        p: The point to find the rightmost point from.
        hull: The convex hull in counter-clockwise order.
    Returns:
        The index rightmost point in `hull` from `p`."""
    if len(hull) == 1:
        return hull[0]

    start = 0
    end = len(hull)
    rightmost = None

    if hull[start] == p:
        start = 1

    while start < end:
        start_prev = find_orientation(p, hull[start],
                                      hull[(end - 1) % len(hull)])
        start_next = find_orientation(p, hull[start],
                                      hull[(start + 1) % len(hull)])

        center = math.floor((start + end) / 2)
        # The direction of the center point relative to the start point
        center_dir = find_orientation(p, hull[start], hull[center])
        center_prev = find_orientation(p, hull[center],
                                       hull[(center - 1) % len(hull)])
        center_next = find_orientation(p, hull[center],
                                       hull[(center + 1) % len(hull)])

        if vplot_is_on(3):
            util.show_plot(hull + [p], hulls=[hull],
                           labels={hull[end % len(hull)]: 'E', hull[start]: 'S',
                                   hull[center]: 'C', p: 'P'})

        if (center_prev in [LEFT_TURN, COLINEAR] and center_next == LEFT_TURN):
            # Our center point is actually the rightmost from p
            rightmost = hull[center]
            break
        elif ((center_dir == LEFT_TURN and
                (start_next == RIGHT_TURN or start_prev == start_next)) or
              (center_dir == RIGHT_TURN and center_prev == RIGHT_TURN) or
              (center_dir == COLINEAR and center_prev == RIGHT_TURN)):
            end = center
        else:
            start = center + 1

    if not rightmost:
        rightmost = hull[start % len(hull)]

    if vplot_is_on(2):
        util.show_plot([p], hulls=[hull], lines=[[p, rightmost]])

    return rightmost


def find_rightmost_in_set(p: Point, candidates: List[Point]) -> Point:
    """Find the point rightmost from p in the given set of candidates.

    Args:
        p: The reference point
        candidates: The set to find the rightmost in.
    Returns:
        The rightmost point from p.
    """
    rightmost = candidates[0]
    for c in candidates:
        if find_orientation(p, rightmost, c) == RIGHT_TURN:
            rightmost = c
    return rightmost


def clockwise_angle_distance(point: Point, origin: Point):
    """
    Adapted from https://www.py4u.net/discuss/183880
    """
    refvec = [0, 1]
    v = point - origin
    lenv = math.hypot(v.x, v.y)
    if lenv == 0:
        return -math.pi, 0
    normalized = Point(v.x/lenv, v.y/lenv)
    dotprod = normalized.x * refvec[0] + normalized.y * refvec[1]
    diffprod = refvec[1] * normalized.x - refvec[0] * normalized.y
    angle = math.atan2(diffprod, dotprod)
    if angle < 0:
        return 2*math.pi + angle, lenv
    return angle, lenv


def b_closer_to_a(a: Point, b: Point, c: Point) -> bool:
    """Adapted from https://github.com/mission-peace/interview/blob/master/src/com/interview/geometry/JarvisMarchConvexHull.java"""
    y1 = a.y - b.y
    y2 = a.y - c.y
    x1 = a.x-b.x
    x2 = a.x - c.x
    return (y1 * y1 + x1 * x1) < (y2 * y2 + x2 * x2)


def y_intesection(a: Point, b: Point, x: float) -> float:
    """Determines the y coordinate of the intersection between the line (`a`,`b`) and the vertical line at x = x`"""
    slope = (b.y - a.y) / (b.x - a.x)
    return a.y + (slope * (x - a.x))


def gift_wrapping(points: List[Point]) -> List[Point]:
    """Implementation of the gift-wrapping algorithm.
    Adapted from https://en.wikipedia.org/wiki/Gift_wrapping_algorithm and
    https://github.com/mission-peace/interview/blob/master/src/com/interview/geometry/JarvisMarchConvexHull.java"""

    hull = []
    sortedPoints = sorted(points, key=lambda p: p.x)
    point = sortedPoints[0]
    first_point_in_hull = point
    secondPoint = points[0]
    hullComplete = False
    collinear_points = set()
    while not hullComplete:
        hull.append(point)
        hull.extend(collinear_points)
        for p in points:
            cross = ((point.y - p.y) * (point.x - secondPoint.x)) - \
                ((point.y - secondPoint.y) * (point.x - p.x))
            p_on_left = cross > 0
            collinear = (cross == 0)
            if p == secondPoint or p == point:
                pass
            elif(point == secondPoint or p_on_left):
                secondPoint = p
                collinear_points = set()
            elif collinear:
                if b_closer_to_a(point, secondPoint, p):
                    collinear_points.append(secondPoint)
                    secondPoint = p
                else:
                    collinear_points.add(p)
        point = secondPoint
        if secondPoint == first_point_in_hull:
            hullComplete = True
    return hull


def divide_and_conquer(points: List[Point]) -> List[Point]:
    if (len(points) < 3):
        return points
    if (len(points) == 3):
        hull = sorted(points, key=lambda p: p.x)
        if (find_orientation(hull[0], hull[1], hull[2]) == CCW):
            hull[1], hull[2] = hull[2], hull[1]
        return hull

    hull = []

    sortedPoints = sorted(points, key=lambda p: p.x)
    half_index = len(points) // 2
    A = sortedPoints[:half_index]
    B = sortedPoints[half_index:]
    ch_A = divide_and_conquer(A)
    ch_B = divide_and_conquer(B)
    r_a = max(ch_A, key=lambda p: p.x)
    l_b = min(ch_B, key=lambda p: p.x)

    # compute upper tangent
    i = ch_A.index(r_a)
    j = ch_B.index(l_b)
    center_x = r_a.x + ((l_b.x - r_a.x) / 2)

    y_IJ = y_intesection(ch_A[i], ch_B[j], center_x)
    y_IJplusone = y_intesection(ch_A[i], ch_B[(j+1) % len(ch_B)], center_x)
    y_IminusoneJ = y_intesection(ch_A[(i-1) % len(ch_A)], ch_B[j], center_x)
    while(y_IJplusone >= y_IJ or y_IminusoneJ >= y_IJ):
        if y_IJplusone >= y_IJ:
            j = (j+1) % len(ch_B)
        else:
            i = (i-1) % len(ch_A)
        y_IJ = y_intesection(ch_A[i], ch_B[j], center_x)
        y_IJplusone = y_intesection(
            ch_A[i], ch_B[(j + 1) % len(ch_B)], center_x)
        y_IminusoneJ = y_intesection(
            ch_A[(i - 1) % len(ch_A)], ch_B[j], center_x)

    upper_a_idx = i
    upper_b_idx = j

    # compute lower tangent
    i = ch_A.index(r_a)
    j = ch_B.index(l_b)
    y_IJ = y_intesection(ch_A[i], ch_B[j], center_x)
    y_IJminusone = y_intesection(ch_A[i], ch_B[(j-1) % len(ch_B)], center_x)
    y_IplusoneJ = y_intesection(ch_A[(i+1) % len(ch_A)], ch_B[j], center_x)
    while (y_IJminusone <= y_IJ or y_IplusoneJ <= y_IJ):
        if y_IJminusone <= y_IJ:
            j = (j-1) % len(ch_B)
        else:
            i = (i + 1) % len(ch_A)
        y_IJ = y_intesection(ch_A[i], ch_B[j], center_x)
        y_IJminusone = y_intesection(
            ch_A[i], ch_B[(j - 1) % len(ch_B)], center_x)
        y_IplusoneJ = y_intesection(
            ch_A[(i + 1) % len(ch_A)], ch_B[j], center_x)

    lower_a_idx = i
    lower_b_idx = j

    # Build the hull
    hull.append(ch_B[upper_b_idx])
    if (upper_b_idx != lower_b_idx):
        i = (upper_b_idx+1) % len(ch_B)
        while (i != lower_b_idx):
            hull.append(ch_B[i])
            i = (i+1) % len(ch_B)
        hull.append(ch_B[lower_b_idx])

    hull.append(ch_A[lower_a_idx])
    if (upper_a_idx != lower_a_idx):
        i = (lower_a_idx+1) % len(ch_A)
        while (i != upper_a_idx):
            hull.append(ch_A[i])
            i = (i+1) % len(ch_A)
        hull.append(ch_A[upper_a_idx])
    return hull


def grahams_scan(points: List[Point]) -> List[Point]:
    """Implementation of Graham's scan."""

    if len(points) < 3:
        return points

    hull = []

    # TODO(havensz@): Add toggle to calculate upper and lower hulls
    # independently using alternating windings instead, as discussed in class.

    p0 = min(points, key=lambda p: (p.y, p.x))
    points = sorted(points, key=lambda p: np.dot(
        p-p0, [1, 0])/np.linalg.norm(p-p0) if p != p0 else 1, reverse=True)

    logging.vlog(2, f"p0: {p0}")
    logging.vlog(2, f"Sorted points: {points}")

    for p in points:
        while len(hull) > 1 and not find_orientation(hull[-2], hull[-1], p) == CCW:
            hull.pop()
        hull.append(p)
        logging.vlog(2, hull)

    return hull


def chans_algorithm(points: List[Point]) -> List[Point]:
    """Implementation of Chan's algorithm."""

    p0 = Point(MAX_COORD, 0)
    p1 = min(points, key=lambda p: p.x)

    if (FLAGS.chans_initial_t < 1):
        raise ValueError()

    # Initial t value can be set via a flag for testing/calibration.
    for t in range(FLAGS.chans_initial_t,
                   max(math.ceil(math.log2(math.log2(len(points))))+1,
                       FLAGS.chans_initial_t + 1)):
        # An estimation for the number of points in the hull using the squaring
        # scheme.
        m = min(2 ** (2 ** t), len(points))
        logging.vlog(1, f"Chan's hull estimation: t={t}, m={m}")

        num_subsets = math.ceil(len(points) / m)
        subset_hulls = []

        # Partition the input set into groups of at most m points and find
        # the convex hull of each.
        for k in range(num_subsets):
            start = k * m
            end = min((k+1)*m, len(points))
            subset_hulls.append(grahams_scan(points[start:end]))

        if vplot_is_on(2):
            util.show_plot(points, hulls=subset_hulls,
                           title=f'Sub-hulls for m={m}')

        # See if we can form a hull with m points or fewer from the subsets
        prev = p0
        curr = p1
        hull = [p1]
        for i in range(m):
            candidates = []
            for k in range(num_subsets):
                candidates.append(find_rightmost_in_hull(curr,
                                  subset_hulls[k]))

            # Find the extreme hull point that maximizes the angle from the
            # last point.
            next = find_rightmost_in_set(curr, candidates)

            if vplot_is_on(1):
                util.show_plot(points, hulls=[hull], lines=[
                               [curr, cand] for cand in candidates],
                               labels={p1: 'p1', curr: 'p', next: 'p+'},
                               title=f"{i}th Point Selection for m={m}")

            if next == p1:
                # We've wound back around to the initial point in less than m
                # hull points, and are done!
                return hull
            else:
                hull.append(next)
                prev = curr
                curr = next

                if FLAGS.chans_eliminate_points:
                    points = [p for hull in subset_hulls for p in hull]

        if vplot_is_on(2):
            util.show_plot(points, hull, label_hulls=True)

    raise LookupError("Hull not found!")


def is_convex(points: List[Point]) -> bool:
    """Determines whether the set of verices forms a convex poly.

    Uses the cross product to determine the angle between adjacent edges. If
    all angles have a consistent winding (left- or right-handed), then the poly
    is convex. A degenerate poly is not considered convex.

    https://en.wikipedia.org/wiki/Cross_product#Computational_geometry

    Args:
        points: The list of vertices that form the polygon.
    Returns:
        Whether the poly is convex.
    """
    if not points or len(points) < 3:
        return False

    winding = None
    for i in range(0, len(points)):
        a = points[i]
        b = points[(i+1) % len(points)]
        c = points[(i+2) % len(points)]
        dir = find_orientation(b, a, c)

        if dir == COLINEAR:
            continue
        elif not winding:
            winding = dir
        elif (dir != winding):
            logging.vlog(
                1, f'Winding is inconsistent between the points {a}->{b}->{c}.')
            if FLAGS.plot_errors:
                util.show_plot(points, lines=[[a, b, c]])
            return False
    return True


def point_in_poly(p1: Point, poly: List[Point]) -> bool:
    """Determines whether the point is in the given polygon.

    Uses a ray-casting method, determining the number of interstions the ray
    has with edges of the poly to indicate whether the point falls within it.
    Intersection testing uses a stripped version of the Bezier-based
    segment-segment equation, as the ray is drawn horizonally, so all
    (p1.y - p2.y) terms will be zero and can be removed.

    https://en.wikipedia.org/wiki/Line-line_intersection#Given_two_points_on_each_line_segment

    Args:
        p1: The point to query.
        poly: An ordered list of vertices that make up the poly.
    Returns:
        Whether the point is within the poly.
    """
    if len(poly) < 3:
        return False

    p2 = Point(sys.maxsize, p1.y)
    num_intersections = 0
    for i in range(len(poly)):
        q1 = poly[i]
        q2 = poly[(i+1) % len(poly)]

        if p1 == q1:
            return True

        denominator = (p1.x - p2.x)*(q1.y - q2.y)
        if denominator == 0:
            if p1.y == q1.y and q1.x <= p1.x <= q2.x:
                # Point lies _on_ a line of the poly.
                return True
            else:
                continue

        t = float((p1.x - q1.x)*(q1.y - q2.y) -
                  (p1.y - q1.y)*(q1.x - q2.x)) / denominator
        u = float(-(p1.y - q1.y)*(p1.x - p2.x)) / denominator

        if (0 <= t <= 1) and (0 <= u <= 1):
            logging.vlog(3, f'PiP intersecton: {p1} -> {q1}-{q2}')
            if t == 0:
                # Point lies _on_ a line of the poly.
                return True
            num_intersections += 1

    return num_intersections % 2 == 1


def validate_hull(hull: List[Point], points: List[Point] = []) -> bool:
    """Validates the convex hull provded as a series of points.

    Checks both that the hull is convex and that all points are either a part of
    the hull or within its confines.

    Args:
        hull: The list of ordered points that make up the convex hull.
        points: Optionally, the full point set the hull was created for
    Returns:
        Whether the hull convex, and is valid for the given points if specified.
    """
    if len(hull) > 3 and not is_convex(hull):
        logging.error("Hull is not convex!")
        return False

    for p in points:
        if not point_in_poly(p, hull):
            logging.error(f"Point {p} does not fall inside the hull!")
            if FLAGS.plot_errors:
                util.show_plot([p], hulls=hull, labels={p: 'p'})
            return False

    return True
