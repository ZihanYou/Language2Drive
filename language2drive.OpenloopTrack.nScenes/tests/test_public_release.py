import shutil
import tempfile
import unittest
from pathlib import Path

from paths import PROJECT_ROOT
from release_io import load_release


class PublicReleaseTest(unittest.TestCase):
    def test_frozen_splits_and_no_private_fields(self):
        release = PROJECT_ROOT / 'releases/nuscenes-test-v1.0.0'
        for split, count in [('all', 75), ('dev', 15), ('heldout', 60)]:
            _, rows = load_release(release, split=split)
            self.assertEqual(len(rows), count)
            for row in rows:
                self.assertNotIn('private_sidecar', row)
                self.assertNotIn('contracts', row)
        _, dev = load_release(release, split='dev')
        _, heldout = load_release(release, split='heldout')
        for key in ('scene_token',):
            self.assertFalse({p['public_input'][key] for p in dev} &
                             {p['public_input'][key] for p in heldout})

    def test_modified_public_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            release = Path(tmp)
            source = PROJECT_ROOT / 'releases/nuscenes-test-v1.0.0'
            for name in ('public_inputs.jsonl', 'release_manifest.json'):
                shutil.copyfile(str(source/name), str(release/name))
            with (release/'public_inputs.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_release(release)
