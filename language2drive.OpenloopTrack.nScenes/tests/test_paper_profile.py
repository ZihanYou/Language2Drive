import json
import tempfile
import unittest
from pathlib import Path

from evaluate_predictions import select_predictions
from model_eval.paper_metrics import (METRICS, collapse_wordings, language_controls,
    score_pair, summarize, paired_completion)
from score_lane_pcus import validate_trajectory


def trajectory(speed):
    return dict(coordinate_frame='ego_at_anchor',reference_point='ego_pose_origin',
                timestamps_s=[i*.25 for i in range(1,13)],
                xy_m=[[speed*i*.25,0.] for i in range(1,13)])


class Map:
    def drivable_at(self,p): return True


def pcu():
    return dict(pcu_id='synthetic',family='scoped_speed',
        public_input={'location':'synthetic','scene_token':'scene','public_input_hash':'hash',
                      'ego_pose':{'xy_m':[0.,0.],'yaw_rad':0.}},
        private_sidecar={'speed_mps':6.,'speed_bands_mps':{'low':[3.2,4.8],'high':[7.2,8.8]}})


def row(condition='canonical',pid='synthetic',c0=1.,c1=1.):
    record=dict(pcu_id=pid,scene_token=pid,family='scoped_speed',condition=condition,
                valid_arms=[True,True],values={k:0. for k in METRICS})
    record['values'].update(paired_completion(c0,c1))
    record['values'].update(SemSISR=(c0+c1)/2,SemCFPS=c0*c1)
    return record


class PaperProfileTest(unittest.TestCase):
    def test_null_string_and_list_predictions_are_invalid(self):
        for value in (None,'unparseable',[],12):
            self.assertIsNotNone(validate_trajectory(value))
            result=score_pair(pcu(),[value,trajectory(8.)],{'synthetic':Map()},{})
            self.assertEqual(result['values']['SemSISR'],.5)
            self.assertEqual(result['values']['SCS'],.5)
            self.assertEqual(result['values']['PCS'],0.)
            self.assertIsNone(result['values']['SRG'])
            self.assertIsNone(result['values']['NBD'])

    def test_paper_scale_and_coverage(self):
        result=score_pair(pcu(),[trajectory(4.),trajectory(8.)],{'synthetic':Map()},{})
        result['condition']='canonical'
        summary=summarize([result],('canonical',))['all']
        for name in ('SemSISR','SemCFPS','SCS','PCS','SRG'):
            self.assertEqual(summary['metrics'][name]['mean'],100.)
        self.assertEqual(summary['coverage']['valid_pairs'],1)
        self.assertEqual(summary['metrics']['paired_ADE_m']['unit'],'m')
        self.assertAlmostEqual(summary['metrics']['NBD']['mean'],100/3)

    def test_empty_valid_response_is_null_not_zero(self):
        result=score_pair(pcu(),[None,None],{'synthetic':Map()},{})
        result['condition']='canonical'
        summary=summarize([result],('canonical',))['all']
        for name in ('SemSISR','SemCFPS','SCS','PCS'):
            self.assertEqual(summary['metrics'][name]['mean'],0.)
            self.assertEqual(summary['metrics'][name]['n'],1)
        for name in ('SRG','NBD','paired_ADE_m'):
            self.assertIsNone(summary['metrics'][name]['mean'])
            self.assertEqual(summary['metrics'][name]['ci95'],[None,None])
            self.assertEqual(summary['metrics'][name]['n'],0)

    def test_pair_before_wording_average(self):
        records=[row('canonical',c0=1,c1=0),row('paraphrase_1',c0=0,c1=1)]
        result=collapse_wordings(records,('canonical','paraphrase_1'))[0]['values']
        self.assertEqual(result['SCS'],.5)
        self.assertEqual(result['PCS'],0.)
        self.assertEqual(result['SemCFPS'],0.)
        with self.assertRaises(ValueError):collapse_wordings(records,('canonical','paraphrase_2'))

    def test_paraphrase_worst_is_pcu_wise(self):
        records=[]
        for i in range(3):
            for j in range(1,4):
                records.append(row('paraphrase_'+str(j),str(i),int(i!=j-1),int(i!=j-1)))
        self.assertEqual(language_controls(records)['Paraphrase_Worst_PCU'],0.)

    def test_control_gaps_retain_negative_values(self):
        records=[row('canonical',c0=0,c1=0),row('masked'),row('shuffled')]
        self.assertEqual(language_controls(records)['masked_CFPS_gap'],-100.)
        self.assertEqual(language_controls(records)['shuffled_CFPS_gap'],-100.)

    def test_eight_speed_intervals_include_one_to_one_quarter(self):
        t=trajectory(4.)
        # Change only the endpoint at 1 s: the following interval is scored.
        t['xy_m'][3][0]=2.
        result=score_pair(pcu(),[t,trajectory(8.)],{'synthetic':Map()},{})
        self.assertEqual(result['own_completion'][0],7/8)
        self.assertEqual(result['binary_arms'][0],0)

    def test_hash_duplicates_and_completeness(self):
        rows=[dict(pcu_id='synthetic',arm=a,condition='canonical',seed=0,
                   public_input_hash='hash',trajectory=trajectory(4.)) for a in ('arm_0','arm_1')]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.jsonl'
            def write(data):path.write_text(''.join(json.dumps(r)+'\n' for r in data))
            write(rows)
            self.assertEqual(len(select_predictions(path,[pcu()],['canonical'],0)),2)
            write(rows[:1])
            with self.assertRaisesRegex(ValueError,'Incomplete'):select_predictions(path,[pcu()],['canonical'],0)
            write(rows+rows[:1])
            with self.assertRaisesRegex(ValueError,'Duplicate'):select_predictions(path,[pcu()],['canonical'],0)
            rows[0]['public_input_hash']='changed';write(rows)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):select_predictions(path,[pcu()],['canonical'],0)
