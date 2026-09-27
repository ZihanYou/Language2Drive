import sys
import unittest
from pathlib import Path
from shapely.geometry import box

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_eval.unified_semantic_metrics import (coverage_matrix_metrics,
    normalized_behavior_difference, progress_factor, compliance, preference_from_distances,
    trajectory_preference)


class UnifiedSemanticTest(unittest.TestCase):
    def test_preference_is_scale_free_and_informative_outside_targets(self):
        self.assertEqual(preference_from_distances(0,1),1)
        self.assertEqual(preference_from_distances(1,0),-1)
        self.assertEqual(preference_from_distances(0,0),0)
        self.assertAlmostEqual(preference_from_distances(3,7),.4)
        self.assertEqual(preference_from_distances(3,7),preference_from_distances(30,70))
        self.assertEqual(preference_from_distances(3,7),-preference_from_distances(7,3))

    def test_reversed_speed_response_detected_outside_both_bands(self):
        p={'family':'scoped_speed','private_sidecar':{'speed_bands_mps':{'low':[3.2,4.8],'high':[7.2,8.8]}}}
        low_prediction={'xy_m':[[12.5*i*.25,0.] for i in range(1,13)]}
        high_prediction={'xy_m':[[0.,0.] for i in range(12)]}
        response=(trajectory_preference(p,low_prediction,{})-trajectory_preference(p,high_prediction,{}))/2
        self.assertLess(response,0)

    def test_correct_and_swapped_semantics(self):
        self.assertEqual(coverage_matrix_metrics([[1,0],[0,1]],True),dict(SCS=1,PCS=1,SRG_coverage=1))
        self.assertEqual(coverage_matrix_metrics([[0,1],[1,0]],True),dict(SCS=0,PCS=0,SRG_coverage=-1))

    def test_instruction_blind_trajectory_has_zero_gain(self):
        r=coverage_matrix_metrics([[.8,.3],[.8,.3]],True)
        self.assertAlmostEqual(r['SRG_coverage'],0)
        self.assertEqual(r['PCS'],.3)

    def test_invalid_pair_is_not_evidence_of_zero_response(self):
        r=coverage_matrix_metrics([[1,0],[0,0]],False)
        self.assertEqual(r['SCS'],.5)
        self.assertEqual(r['PCS'],0)
        self.assertIsNone(r['SRG_coverage'])

    def test_behavior_change_does_not_imply_correctness(self):
        a,b=[[1.,0.],[2.,0.]],[[2.,0.],[4.,0.]]
        d=normalized_behavior_difference(a,b)
        self.assertAlmostEqual(d,1/3)
        self.assertEqual(d,normalized_behavior_difference(b,a))
        self.assertEqual(normalized_behavior_difference(a,a),0)
        self.assertEqual(normalized_behavior_difference([[0.,0.]],[[0.,0.]]),0)
        self.assertEqual(normalized_behavior_difference([[1.,0.]],[[-1.,0.]]),1)

    def test_progress_soft_gate(self):
        self.assertEqual(progress_factor([[0.,0.]]),0)
        self.assertEqual(progress_factor([[4.,0.]]),.5)
        self.assertEqual(progress_factor([[10.,0.]]),1)

    def test_lane_transition_not_counted_as_opposite_response(self):
        times=[i*.25 for i in range(1,13)]
        t={'coordinate_frame':'ego_at_anchor','reference_point':'ego_pose_origin',
           'timestamps_s':times,'xy_m':[[10*x,0. if x<2.25 else 3.] for x in times]}
        p={'pcu_id':'test','family':'lane_change_keep',
           'public_input':{'ego_pose':{'xy_m':[0.,0.],'yaw_rad':0.}}}
        c={'test':(box(-1,-1,50,1),box(-1,2,50,4))}
        self.assertAlmostEqual(compliance(p,t,'arm_1',c),8/12)
        self.assertEqual(compliance(p,t,'arm_1',c,shared_scope=True),0)
        self.assertEqual(compliance(p,t,'arm_0',c,shared_scope=True),1)


if __name__=='__main__':
    unittest.main()
