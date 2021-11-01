import convex_hull

from typing import List, Tuple

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from point import Point


def make_point_set(coords: List[Tuple[int, int]]):
    return [Point(a[0], a[1]) for a in coords]


class UtilityUnitTest(absltest.TestCase):
    def testPiPNoPoly(self):
        self.assertFalse(convex_hull.point_in_poly(Point(0, 0), []))

    def testPiPOnePolyPoint(self):
        self.assertFalse(convex_hull.point_in_poly(Point(0, 0), [Point(1, 1)]))

    def testPiPNoIntersection(self):
        p = Point(-1, 1)
        poly = make_point_set([(0, 0), (0, 2), (2, 2), (2, 0)])
        self.assertFalse(convex_hull.point_in_poly(p, poly))

    def testPiPOnAxisLine(self):
        p = Point(1, 2)
        poly = make_point_set([(0, 0), (0, 2), (2, 2), (2, 0)])
        self.assertTrue(convex_hull.point_in_poly(Point(0, 1), poly))
        self.assertTrue(convex_hull.point_in_poly(Point(1, 0), poly))
        self.assertTrue(convex_hull.point_in_poly(Point(2, 1), poly))
        self.assertTrue(convex_hull.point_in_poly(Point(1, 2), poly))

    def testPiPOnNonAxisLine(self):
        poly = make_point_set([(1, 0), (0, 1), (1, 2), (2, 1)])

        self.assertTrue(convex_hull.point_in_poly(Point(1.5, 1.5), poly))
        self.assertTrue(convex_hull.point_in_poly(Point(1.5, 0.5), poly))
        self.assertTrue(convex_hull.point_in_poly(Point(0.5, 1.5), poly))
        self.assertTrue(convex_hull.point_in_poly(Point(0.5, 0.5), poly))

    def testPiPInSquare(self):
        p = Point(1, 1)

        poly = make_point_set([(0, 0), (0, 2), (2, 2), (2, 0)])
        self.assertTrue(convex_hull.point_in_poly(p, poly))

    def testPiPEvenIntersections(self):
        p = Point(-1, 1)
        poly = make_point_set([(0, 0), (0, 2), (2, 2), (2, 0)])
        self.assertFalse(convex_hull.point_in_poly(p, poly))

    def testPiPIsPoint(self):
        poly = make_point_set([(1, 0), (0, 1), (1, 2), (2, 1)])

        self.assertTrue(convex_hull.point_in_poly(Point(1, 0), poly))
        self.assertTrue(convex_hull.point_in_poly(Point(0, 1), poly))
        self.assertTrue(convex_hull.point_in_poly(Point(1, 2), poly))
        self.assertTrue(convex_hull.point_in_poly(Point(2, 1), poly))

    def testIsConvexEmptyHull(self):
        self.assertFalse(convex_hull.is_convex([]))

    def testIsConvexOnePoint(self):
        self.assertFalse(convex_hull.is_convex([Point(0, 0)]))

    def testIsConvexTwoPoints(self):
        self.assertFalse(convex_hull.is_convex([Point(0, 0), Point(1, 1)]))

    def testIsConvexTrue(self):
        self.assertTrue(convex_hull.is_convex(
            make_point_set([(0, 0), (2, 0), (2, 2), (0, 2)])))

    def testIsConvexFalse(self):
        self.assertFalse(convex_hull.is_convex(
            make_point_set([(0, 0), (2, 0), (2, 2), (1, 1), (0, 2)])))


@ parameterized.parameters(
    convex_hull.gift_wrapping,
    convex_hull.divide_and_conquer,
    convex_hull.grahams_scan,
    convex_hull.chans_algorithm
)
class HullAlgorithmTest(parameterized.TestCase):
    def testSimpleHull(self, algo):
        pass


if __name__ == '__main__':
    logging.set_verbosity(2)
    absltest.main()
