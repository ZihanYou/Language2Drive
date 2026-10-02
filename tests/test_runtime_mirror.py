"""Mirror provisioning must preserve frozen versions, markers and hashes."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mirror_pins', ROOT / 'language2drive.RealSim/scripts/prepare_runtime_mirror.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MirrorTests(unittest.TestCase):
    def test_registry_pins_are_verbatim_and_direct_sources_stay_native(self):
        registry = 'numpy==2.1.3 ; python_version >= "3.12" \\\n    --hash=sha256:' + 'a' * 64 + '\n'
        source = 'local @ git+https://example.com/repo.git@abc\n' + registry + 'archive @ https://example.com/a.whl \\\n    --hash=sha256:' + 'b' * 64 + '\n'
        self.assertEqual(MODULE.registry_requirements(source), registry)

    def test_unhashed_registry_pin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no frozen artifact hashes'):
            MODULE.registry_requirements('numpy==2.1.3\n')
