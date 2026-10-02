"""Process ownership, frozen-group coverage and complete-campaign validation."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'language2drive.RealSim/src'))
from l2d_benchmark.cli import selection
from l2d_benchmark.configure import configure
from l2d_benchmark.live import launch, check
from l2d_benchmark.processes import Services, listening
from l2d_benchmark.realsim_runtime import frozen_groups
from l2d_benchmark.setup import transport_environment


class LiveTests(unittest.TestCase):
    def test_broken_proxy_falls_back_without_forwarding_proxy_setting(self):
        network = {'PIP_INDEX_URL': 'https://pypi.org/simple', 'HF_ENDPOINT': 'https://huggingface.co'}
        with patch.dict(os.environ, {'HTTPS_PROXY': 'http://unreachable.example:80'}), patch('l2d_benchmark.setup.probe', side_effect=[False, True]):
            env, mode = transport_environment(network)
        self.assertEqual(mode, 'direct')
        self.assertNotIn('HTTPS_PROXY', env)
        self.assertEqual(env['NO_PROXY'], '*')

    def test_three_groups_cover_all_five(self):
        self.assertEqual(selection('carla,realsim,openloop'), selection('all'))
        with self.assertRaises(argparse.ArgumentTypeError):
            selection('navsim,openloop')

    def test_configure_and_dry_plan_preserve_all_five(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / 'eval.json'
            configure(ROOT, config)
            report = launch(config, selection('all'), Path(temp) / 'out', ROOT / 'examples/constant_velocity.py', dry_run=True)
            self.assertEqual(set(report['tracks']), set(selection('all')))
            self.assertEqual(report['runtime_groups']['openloop'], ['nuscenes', 'navsim', 'wod'])
            self.assertIn('/.runtime-env/venv/bin/python', report['tracks']['realsim']['commands'][0][0])
            command = report['tracks']['carla']['commands'][0]
            self.assertEqual(command[command.index('--gpu-rank') + 1], '0')
            self.assertFalse((Path(temp) / 'out').exists())
            with self.assertRaises(ValueError):
                configure(ROOT, config)
            custom = Path(temp) / 'local-disk.json'
            configure(ROOT, custom, realsim_work_dir=Path(temp) / 'runtime')
            custom_plan = launch(custom, ['realsim'], Path(temp) / 'out',
                                 ROOT / 'examples/constant_velocity.py', dry_run=True)
            self.assertEqual(custom_plan['tracks']['realsim']['commands'][0][0],
                             str(Path(temp) / 'runtime/venv/bin/python'))

    def test_preflight_does_not_claim_experiments(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / 'eval.json'; configure(ROOT, config)
            with patch('l2d_benchmark.live.subprocess.run', side_effect=OSError('driver hidden')):
                report = check(config, ['navsim'])
            self.assertFalse(report['ready'])
            self.assertFalse(report['gpu_experiments_executed'])
            self.assertEqual(report['gpu_inventory'], [])

    def test_owned_service_cleanup_and_occupied_port_preservation(self):
        with tempfile.TemporaryDirectory() as temp, socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]; sock.close()
            with Services(temp) as services:
                process = services.start('http', [sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'], temp, os.environ.copy(), port, 10)
                self.assertTrue(listening('127.0.0.1', port))
                with self.assertRaisesRegex(RuntimeError, 'already in use'):
                    services.start('collision', [sys.executable, '-c', 'raise SystemExit(0)'], temp, os.environ.copy(), port, 1)
                self.assertIsNone(process.poll())
            self.assertIsNotNone(process.poll())
            self.assertFalse(listening('127.0.0.1', port))

    def test_failed_service_retains_log_and_is_cleaned(self):
        with tempfile.TemporaryDirectory() as temp, socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]; sock.close()
            with self.assertRaisesRegex(RuntimeError, 'exited'):
                with Services(temp) as services:
                    services.start('failure', [sys.executable, '-c', 'print("startup failure", flush=True); raise SystemExit(7)'], temp, os.environ.copy(), port, 5)
            self.assertIn('startup failure', (Path(temp) / 'failure.log').read_text())

    def test_frozen_group_exact_coverage(self):
        from language2drive_realsim.alpasim_bridge import ALPASIM_COMMIT
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'runtime').mkdir()
            (root / 'manifest.json').write_text(json.dumps(dict(alpasim_commit=ALPASIM_COMMIT, seed=1701, scheduled_slots=60)))
            ids = [f'pcu-{i}' for i in range(30)]
            plan = dict(runs=[dict(pcu_id=p) for p in ids for _ in range(2)])
            groups = [dict(id='group-00', pcu_ids=ids[:15]), dict(id='group-01', pcu_ids=ids[15:])]
            path = root / 'runtime/groups.json'; path.write_text(json.dumps(groups))
            self.assertEqual(len(frozen_groups(root, plan)), 2)
            groups[1]['pcu_ids'].append(ids[0]); path.write_text(json.dumps(groups))
            with self.assertRaisesRegex(ValueError, 'repeated'):
                frozen_groups(root, plan)

    def test_completeness_rejects_empty_and_partial_campaign(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('validate_campaign', ROOT / 'scripts/validate_gpu_campaign.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'summary.json').write_text(json.dumps(dict(tracks={})))
            report = module.validate(root)
            self.assertFalse(report['complete'])
            self.assertFalse(report['formal_submission_approved'])
            self.assertGreaterEqual(len(report['failures']), 5)

    def test_scheduled_carla_runs_without_valid_scores_are_incomplete(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('validate_campaign', ROOT / 'scripts/validate_gpu_campaign.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'summary.json').write_text(json.dumps(dict(tracks={})))
            for split, count in [('dev', 96), ('test', 288)]:
                target = root / 'carla' / split; target.mkdir(parents=True)
                (target / 'run_plan.json').write_text(json.dumps(dict(evaluation_scope={'run_ids': list(range(count))})))
                (target / 'score.json').write_text(json.dumps(dict(metrics={'expected_runs': count, 'valid_result_runs': 0, 'finalized_runs': 0})))
            report = module.validate(root)
            self.assertEqual(report['counts']['carla'], 384)
            self.assertFalse(report['complete'])
            self.assertTrue(any('unbound dev' in error for error in report['failures']))
