import unittest
from score_lane_pcus import score_one


class Geometry:
    def drivable_at(self, point):
        return True


def trajectory(speed):
    return dict(coordinate_frame='ego_at_anchor', reference_point='ego_pose_origin',
                timestamps_s=[.25*i for i in range(1,13)],
                xy_m=[[speed*.25*i,0.] for i in range(1,13)])


class BinaryScoringTest(unittest.TestCase):
    def setUp(self):
        self.pcu = dict(family='scoped_speed',
            public_input={'ego_pose': {'xy_m':[0.,0.], 'yaw_rad':0.}},
            private_sidecar={'speed_mps':6., 'speed_bands_mps':
                             {'low':[3.2,4.8], 'high':[7.2,8.8]}})

    def test_correct_and_opposite_contracts(self):
        own=score_one(Geometry(),self.pcu,'arm_0',trajectory(4.))
        other=score_one(Geometry(),self.pcu,'arm_1',trajectory(4.))
        self.assertTrue(own['semantic_success'])
        self.assertFalse(other['semantic_success'])
        self.assertEqual(other['failure_code'],'O2_WRONG_SEMANTIC')

    def test_no_commit_and_invalid_are_failures(self):
        stopped=score_one(Geometry(),self.pcu,'arm_0',trajectory(0.))
        self.assertEqual(stopped['failure_code'],'O1_NO_COMMIT')
        bad=trajectory(4.)
        bad['xy_m'][0][0]=float('nan')
        invalid=score_one(Geometry(),self.pcu,'arm_0',bad)
        self.assertFalse(invalid['semantic_success'])
        self.assertFalse(invalid['output_valid'])

    def test_semantics_and_motion_are_separate(self):
        self.pcu['private_sidecar']['speed_mps']=0.
        result=score_one(Geometry(),self.pcu,'arm_1',trajectory(8.))
        self.assertTrue(result['semantic_success'])
        self.assertFalse(result['trajectory_feasible'])
