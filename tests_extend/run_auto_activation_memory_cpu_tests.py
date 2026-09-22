import argparse
import hashlib
import importlib
import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'mindspeed.core.pipeline_parallel.adaptive_offload'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0'
    if options.output is not None:
        assert not options.output.exists(), str(options.output)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    import torch
    import torch_npu
    from torch.utils.checkpoint import DefaultDeviceType

    assert not torch.cuda.is_initialized() and not torch_npu.npu.is_initialized()
    original_device = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type('cpu')
    try:
        with patch.object(torch_npu.npu, '_lazy_init', side_effect=AssertionError('CPU tests initialized NPU')) as lazy_guard, \
                patch.object(torch_npu._C, '_npu_init', side_effect=AssertionError('CPU tests initialized native NPU')) as native_guard:
            suite = unittest.defaultTestLoader.discover(
                str(ROOT / 'tests_extend/unit_tests/mindspeed/core/pipeline_parallel'), pattern='test_*.py')
            expected = suite.countTestCases()
            assert expected >= 354, expected
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert lazy_guard.call_count == native_guard.call_count == 0
        assert DefaultDeviceType.get_device_type() == 'cpu'
    finally:
        DefaultDeviceType.set_device_type(original_device)
    assert not torch.cuda.is_initialized() and not torch_npu.npu.is_initialized()
    bindings = {}
    for name, module in tuple(sys.modules.items()):
        if name == PREFIX or name.startswith(PREFIX + '.') or name == 'mindspeed.core.megatron_basic.requirements_basic':
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(ROOT / 'mindspeed'), (name, str(path))
            bindings[name] = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ('auto_activation_memory', 'dense_activation_boundary', 'bandwidth_calibration', 'transfer_cost_feedback'):
        importlib.import_module(PREFIX + '.' + name)
        assert PREFIX + '.' + name in bindings
    report = {'timestamp': datetime.now().astimezone().isoformat(),
              'passed': result.wasSuccessful() and result.testsRun == expected and not result.skipped,
              'tests_run': result.testsRun, 'expected_tests': expected, 'failures': len(result.failures),
              'errors': len(result.errors), 'skipped': len(result.skipped), 'bindings': bindings,
              'npu_initialized': False, 'cuda_initialized': False, 'source_overlay_used': False,
              'checkpoint_default_restored': DefaultDeviceType.get_device_type() == original_device,
              'native_training_rerun': False}
    if options.output is not None:
        with options.output.open('x') as output:
            json.dump(report, output, indent=2)
            output.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'bindings'}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
