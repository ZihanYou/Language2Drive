import sys
import unittest
from pathlib import Path

from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_eval.semantic_metrics import (TIMES, ade, aggregate, corridor_stats,
                                         lane_preference, paired_response, speed_stats)


class SemanticMetricsTest(unittest.TestCase):
    def test_speed_target_and_temporal_failure_are_distinct(self):
        xy = [[15 / 3.6 * t, 0.] for t in TIMES]
        good = speed_stats(xy, [3.2, 4.8], 15)
        self.assertAlmostEqual(good['speed_target_mae_kmh'], 0.)
        self.assertEqual(good['speed_coverage'], 1.)
        self.assertGreater(good['speed_robustness_kmh'], 0.)
        xy[-1][0] += 1.
        brief_failure = speed_stats(xy, [3.2, 4.8], 15)
        self.assertAlmostEqual(brief_failure['speed_coverage'], 7 / 8)
        self.assertLess(brief_failure['speed_robustness_kmh'], 0.)

    def test_speed_response_preserves_sign(self):
        low, high = {'speed_mean_kmh': 15}, {'speed_mean_kmh': 30}
        self.assertEqual(paired_response('scoped_speed', low, high), 15)
        self.assertEqual(paired_response('scoped_speed', high, low), -15)
        self.assertEqual(paired_response('scoped_speed', low, low), 0)

    def test_map_corridor_and_response(self):
        current, target = box(-1, -1, 50, 1), box(-1, 2, 50, 4)
        keep = [[10*t, 0.] for t in TIMES]
        change = [[10*t, 3.] for t in TIMES]
        self.assertEqual(corridor_stats(change, target, 2.25)['lane_coverage'], 1)
        self.assertEqual(corridor_stats(keep, target, 2.25)['lane_outside_m'], 2)
        a = {'lane_preference_m': lane_preference(change, current, target)}
        b = {'lane_preference_m': lane_preference(keep, current, target)}
        self.assertEqual(paired_response('lane_change_keep', a, b), 4)
        self.assertEqual(paired_response('lane_change_keep', b, a), -4)
        self.assertEqual(paired_response('lane_change_keep', b, b), 0)

    def test_lane_hold_window(self):
        points = [[t*10, 0. if t < 2.25 else 3.] for t in TIMES]
        corridor = box(-1, 2, 50, 4)
        self.assertEqual(corridor_stats(points, corridor, 2.25)['lane_coverage'], 1.)
        self.assertAlmostEqual(corridor_stats(points, corridor, 0.)['lane_coverage'], 4/12)

    def test_ade_cannot_determine_semantic_direction(self):
        a, b = [[1., 0.]], [[1., 3.]]
        self.assertEqual(ade(a, b), ade(b, a))
        self.assertEqual(ade(a, a), 0.)

    def test_cluster_mean_keeps_valid_measurement_denominator(self):
        result = aggregate([('a', 1.), ('a', 3.), ('b', 5.)], ['a', 'b', 'c'])
        self.assertEqual(result['mean'], 3.)
        self.assertEqual(result['n'], 3)
        self.assertEqual(aggregate([], ['a'])['mean'], None)


if __name__ == '__main__':
    unittest.main()
