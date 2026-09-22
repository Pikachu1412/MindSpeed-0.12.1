import hashlib
import tempfile
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from mindspeed.core.pipeline_parallel.adaptive_offload import auto_activation_memory as automatic
from mindspeed.core.megatron_basic import requirements_basic as compatibility


CORE = 'mindspeed.core.megatron_basic.requirements_basic'
REGISTRATION = 'mindspeed.features_manager.megatron_basic.requirements_basic'
SOURCE = Path(compatibility.__file__).resolve()


def module_at(name, path):
    module = ModuleType(name)
    module.__file__ = str(path)
    return module


class RuntimeSourceFingerprintTest(unittest.TestCase):
    def fingerprint(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            return automatic.profile_fingerprint([torch.nn.Linear(2, 2)], SimpleNamespace())

    def test_exact_loaded_source_content_is_hashed(self):
        path = SOURCE
        with patch.dict(sys.modules, {CORE: module_at(CORE, path), REGISTRATION: None}):
            dependencies = automatic._profile_runtime_sources()
            self.assertEqual(dependencies[CORE], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertIsNone(dependencies[REGISTRATION])

    def test_live_and_fixed_compatibility_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            fingerprints = []
            for version in (1, 2):
                path = Path(directory) / f'compatibility_{version}.py'
                path.write_text(SOURCE.read_text() + f'\nFINGERPRINT_TEST_VERSION = {version}\n')
                with patch.dict(sys.modules, {CORE: module_at(CORE, path), REGISTRATION: None}):
                    fingerprints.append(self.fingerprint())
            self.assertNotEqual(*fingerprints)

    def test_same_path_source_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runtime.py'
            path.write_text('VERSION = 1\n')
            with patch.dict(sys.modules, {CORE: module_at(CORE, path), REGISTRATION: None}):
                before = self.fingerprint()
                path.write_text('VERSION = 2\n')
                self.assertNotEqual(before, self.fingerprint())

    def test_same_content_at_other_path_preserves_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            fingerprints = []
            for filename in ('first.py', 'second.py'):
                path = Path(directory) / filename
                path.write_text('VERSION = 1\n')
                with patch.dict(sys.modules, {CORE: module_at(CORE, path), REGISTRATION: None}):
                    fingerprints.append(self.fingerprint())
            self.assertEqual(*fingerprints)

    def test_absent_and_loaded_dependencies_do_not_share_signature(self):
        with patch.dict(sys.modules, {CORE: None, REGISTRATION: None}):
            absent = self.fingerprint()
            path = SOURCE
            sys.modules[CORE] = module_at(CORE, path)
            self.assertNotEqual(absent, self.fingerprint())

    def test_registration_changes_invalidate_signature(self):
        path = SOURCE
        with patch.dict(sys.modules, {CORE: module_at(CORE, path), REGISTRATION: None}):
            absent = self.fingerprint()
            sys.modules[REGISTRATION] = module_at(REGISTRATION, path)
            self.assertNotEqual(absent, self.fingerprint())

    def test_unreadable_loaded_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.py'
            with patch.dict(sys.modules, {CORE: module_at(CORE, path), REGISTRATION: None}):
                with self.assertRaises((OSError, TypeError)):
                    self.fingerprint()

    def test_unrelated_loaded_module_does_not_change_signature(self):
        path = SOURCE
        with patch.dict(sys.modules, {CORE: module_at(CORE, path), REGISTRATION: None}):
            before = self.fingerprint()
            with patch.dict(sys.modules, {'unrelated_fingerprint_fixture': ModuleType('unrelated_fingerprint_fixture')}):
                self.assertEqual(before, self.fingerprint())
