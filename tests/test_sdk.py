"""Protocol regressions: model boundary, resume identity and native exports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from l2d_benchmark.adapter import Request, SPECS, load_adapter, predict, validate_trajectory
from l2d_benchmark.runners import queries, export, run_openloop
from l2d_benchmark.setup import sources, bootstrap
from l2d_benchmark.pipeline import run


class SDKTests(unittest.TestCase):
    def test_all_five_requested_grids_and_invalid_outputs(self):
        adapter = load_adapter(ROOT / 'examples/constant_velocity.py')
        for track, spec in SPECS.items():
            request = Request(track, 'Maintain speed.', {}, spec)
            result = predict(adapter, request)
            self.assertEqual(len(result['xy_m']), len(spec.timestamps_s))
            for field, value in [('reference_point', 'unknown'), ('timestamps_s', [True] * len(spec.timestamps_s)), ('xy_m', [[float('nan'), 0]] * len(spec.timestamps_s))]:
                with self.assertRaises(ValueError):
                    validate_trajectory(dict(result, **{field: value}), spec)

    def test_factory_creates_distinct_session_state(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = Path(temp) / 'model.py'
            adapter.write_text('class Adapter:\n    def predict(self, request): pass\ndef create_adapter(config): return Adapter()\n')
            self.assertIsNot(load_adapter(adapter), load_adapter(adapter))

    def test_actual_http_service_validates_all_tracks(self):
        import os
        import socket
        import subprocess
        import time
        from l2d_benchmark.service import Client
        from urllib.error import HTTPError
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        process = subprocess.Popen([sys.executable, '-m', 'l2d_benchmark.cli', 'serve', '--adapter', str(ROOT / 'examples/constant_velocity.py'), '--port', str(port)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            client = Client(f'http://127.0.0.1:{port}', timeout=2)
            deadline = time.monotonic() + 10
            while True:
                try:
                    identity = client.identity()
                    break
                except OSError:
                    if time.monotonic() >= deadline or process.poll() is not None:
                        self.fail('Model service failed to start')
                    time.sleep(.05)
            self.assertEqual(len(identity['adapter_sha256']), 64)
            for track, spec in SPECS.items():
                request = Request(track, 'Maintain speed.', {}, spec)
                with self.assertRaisesRegex(ValueError, "Reset the session") :
                    client.predict(request, track)
                client.reset(track)
                result = client.predict(request, track)
                self.assertEqual(len(result['xy_m']), len(spec.timestamps_s))
                client.call('close', {'session': track})
            # Exercise the real coordinator -> worker -> image transport ->
            # native scorer chain with a synthetic, explicitly non-paper PCU.
            from PIL import Image
            with tempfile.TemporaryDirectory() as temp:
                fixture = Path(temp)
                public = fixture / 'public'
                (public / 'public_inputs').mkdir(parents=True)
                Image.new('RGB', (8, 8), 'white').save(fixture / 'camera.png')
                (public / 'manifest.json').write_text(json.dumps({'pcu_ids': ['toy_speed']}))
                row = {'instructions': ['Drive at 4.5 m/s.', 'Drive at 7.5 m/s.'], 'paraphrases': [['Maintain 4.5 m/s.'] * 3, ['Maintain 7.5 m/s.'] * 3], 'camera_paths_history': [{'CAM_FRONT': 'camera.png'}] * 4, 'ego_history': []}
                (public / 'public_inputs/toy_speed.json').write_text(json.dumps(row))
                config = {'workspace': str(ROOT), 'endpoint': client.endpoint, 'model': {'name': 'constant-speed-fixture'}, 'tracks': {'navsim': {'python': sys.executable, 'public_dir': str(public), 'sensor_root': str(fixture), 'sidecars': str(ROOT / 'language2drive.OpenloopTrack.navsim/examples/toy_sidecars')}}}
                path = fixture / 'evaluation.json'
                path.write_text(json.dumps(config))
                output = fixture / 'results'
                report = run(path, ['navsim'], output)
                self.assertEqual(report['tracks']['navsim']['status'], 'scored', report)
                inference = json.loads((output / 'navsim/inference.json').read_text())
                self.assertEqual(inference['scheduled_requests'], 8)
                self.assertEqual(inference['failed_requests'], 0)
                score = json.loads((output / 'navsim/score.json').read_text())
                self.assertEqual(score['ol_sem_cfps'], 0)
                self.assertEqual(score['trajectory_valid_rate'], 1)
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_public_navsim_boundary_all_wordings(self):
        backend = ROOT / 'language2drive.OpenloopTrack.navsim'
        with patch('l2d_benchmark.runners.image', return_value={'encoding': 'png_base64', 'data': ''}):
            requests = list(queries('navsim', backend, {'sensor_root': '/unused'}))
        self.assertEqual(len(requests), 38 * 2 * 4)
        for identity, request in requests:
            self.assertEqual(set(request.observation), {'cameras', 'ego_history'})
            serialized = json.dumps(request.to_dict())
            self.assertNotIn(identity['pcu_id'], serialized)
            self.assertNotIn('balanced_pre_review', serialized)
            self.assertIsInstance(request.instruction, str)

    def test_nuscenes_public_boundary_and_native_grid(self):
        backend = ROOT / 'language2drive.OpenloopTrack.nScenes'
        with patch('l2d_benchmark.runners.image', return_value={'encoding': 'png_base64', 'data': ''}):
            requests = list(queries('nuscenes', backend, {'sensor_root': '/unused'}))
        self.assertEqual(len(requests), 120)
        for identity, request in requests:
            serialized = json.dumps(request.to_dict())
            self.assertNotIn(identity['pcu_id'], serialized)
            self.assertNotIn('private_contract', serialized)
            self.assertEqual(len(request.output_spec.timestamps_s), 12)

    def test_resume_and_changed_model_refusal(self):
        request = Request('navsim', 'Continue.', {}, SPECS['navsim'])
        meta = {'pcu_id': 'toy', 'arm': '0', 'condition': 'canonical', 'seed': 0}
        result = predict(load_adapter(ROOT / 'examples/constant_velocity.py'), request)
        with tempfile.TemporaryDirectory() as temp, patch('l2d_benchmark.runners.Client') as client, patch('l2d_benchmark.runners.queries', return_value=[(meta, request)]):
            client.return_value.identity.return_value = {'adapter_sha256': 'abc'}
            client.return_value.predict.return_value = result
            first = run_openloop('navsim', ROOT, {}, temp, 'http://unused', {'name': 'same'})
            second = run_openloop('navsim', ROOT, {}, temp, 'http://unused', {'name': 'same'})
            self.assertEqual(first, second)
            self.assertEqual(client.return_value.predict.call_count, 1)
            with self.assertRaises(ValueError):
                run_openloop('navsim', ROOT, {}, temp, 'http://unused', {'name': 'changed'})

    def test_failures_remain_in_scheduled_inventory(self):
        request = Request('wod', 'Stop.', {}, SPECS['wod'])
        with tempfile.TemporaryDirectory() as temp, patch('l2d_benchmark.runners.Client') as client, patch('l2d_benchmark.runners.queries', return_value=[({'request_id': 'scheduled'}, request)]):
            client.return_value.identity.return_value = {}
            client.return_value.predict.side_effect = ValueError('bad model output')
            report = run_openloop('wod', ROOT, {}, temp, 'http://unused', {})
            self.assertEqual(report['failed_requests'], 1)
            row = json.loads((Path(temp) / 'predictions.jsonl').read_text())
            self.assertEqual(row, {'request_id': 'scheduled', 'trajectory': []})

    def test_source_selection_and_standalone_carla(self):
        with patch('l2d_benchmark.setup.reachable', side_effect=[False, True]):
            self.assertEqual(sources(), {'HF_ENDPOINT': 'https://hf-mirror.com', 'PIP_INDEX_URL': 'https://pypi.org/simple'})
        is_file = Path.is_file
        with patch.object(Path, 'is_file', lambda path: False if str(path).endswith('/.l2d/envs/carla/bin/python') else is_file(path)):
            plan = bootstrap(ROOT, ['carla', 'realsim'], mode='official', runtime=True, dry_run=True)
        self.assertIn('create', plan['commands'][0]['argv'])
        self.assertIn('python=3.11', plan['commands'][0]['argv'])
        self.assertTrue(any('setup_runtime.py' in ' '.join(s['argv']) for s in plan['commands']))
        native_sdk = next(s['argv'] for s in plan['commands']
                          if s['track'] == 'realsim' and '--no-deps' in s['argv']
                          and '--python' in s['argv'])
        self.assertTrue(native_sdk[0].endswith('/.runtime-env/bootstrap/bin/uv'))
        self.assertIn('--no-config', native_sdk)
        self.assertTrue(native_sdk[native_sdk.index('--python') + 1].endswith('/.runtime-env/venv/bin/python'))
        custom = bootstrap(ROOT, ['realsim'], mode='official', runtime=True,
                           dry_run=True, realsim_work_dir='/tmp/l2d-test-runtime')
        native_setup = next(s['argv'] for s in custom['commands'] if 'setup_runtime.py' in ' '.join(s['argv']))
        self.assertEqual(native_setup[native_setup.index('--work-dir') + 1], '/tmp/l2d-test-runtime')
        self.assertTrue(any(s['argv'][0] == '/tmp/l2d-test-runtime/bootstrap/bin/uv'
                            and '/tmp/l2d-test-runtime/venv/bin/python' in s['argv'] for s in custom['commands']))
        mirrored = bootstrap(ROOT, ['nuscenes'], mode='mirror', download=True, dry_run=True)
        self.assertEqual(len(mirrored['resources_pending']), 1)

    def test_pipeline_dry_run_all_tracks_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results'
            report = run(ROOT / 'examples/evaluation.json', list(SPECS), output, dry_run=True)
            self.assertEqual(set(report['tracks']), set(SPECS))
            self.assertFalse(output.exists())

    def test_native_navsim_accepts_unified_export(self):
        import subprocess
        backend = ROOT / 'language2drive.OpenloopTrack.navsim'
        canonical = json.loads((backend / 'examples/toy_predictions.json').read_text())
        rows = []
        for pcu, arms in canonical.items():
            for arm, trajectory in arms.items():
                rows.append({'meta': {'pcu_id': pcu, 'arm': arm, 'condition': 'canonical'}, 'trajectory': {'coordinate_frame': trajectory['coordinate_frame'], 'reference_point': trajectory['reference_point'], 'timestamps_s': trajectory['timestamps'], 'xy_m': trajectory['positions_xy']}})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            export('navsim', rows, output)
            self.assertEqual(json.loads((output / 'canonical.json').read_text()), {p: {a: {k: v for k, v in t.items() if k != 'headings'} for a, t in arms.items()} for p, arms in canonical.items()})
            subprocess.run([sys.executable, str(backend / 'navsim_offline/score_corrected.py'), '--sidecars', str(backend / 'examples/toy_sidecars'), '--predictions', str(output / 'canonical.json'), '--output', str(output / 'score.json'), '--bootstrap-samples', '10'], check=True, capture_output=True)
            self.assertTrue((output / 'score.json').exists())


if __name__ == '__main__':
    unittest.main()
