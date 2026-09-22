import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PROJECT = REPOSITORY.parent
MEGATRON = PROJECT / 'Megatron-LM-0.12.1'
LAUNCHER = MEGATRON / 'run_offload.sh'
CASES = (('cold', 80), ('warm', 20))
DEVICE_PATTERN = re.compile(r'^\|\s*([0-3])\s+0\s*\|\s*(\d+)\s*\|', re.M)
STEP_PATTERN = re.compile(r'iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+).*?lm loss:\s*(\S+).*?grad norm:\s*(\S+).*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now().astimezone().isoformat()


def process_state(pid):
    directory = Path('/proc') / str(pid)
    try:
        fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': pid, 'parent': int(fields[1]), 'start_ticks': fields[19], 'uid': directory.stat().st_uid}
    except FileNotFoundError:
        return None


def descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        state = process_state(pid)
        if state is None:
            return False
        pid = state['parent']
    return False


def devices():
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True, timeout=20)
    return {int(device): int(pid) for device, pid in DEVICE_PATTERN.findall(result.stdout)}


def environment(directory, iterations):
    result = dict(os.environ)
    result.update({'PATH': str(Path(sys.executable).parent) + ':' + os.environ['PATH'],
                   'PYTHONPATH': str(REPOSITORY) + ':' + str(MEGATRON), 'PYTHONDONTWRITEBYTECODE': '1',
                   'TORCH_DEVICE_BACKEND_AUTOLOAD': '0', 'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3',
                   'GPUS_PER_NODE': '4', 'PIPELINE_PARALLEL_SIZE': '4', 'MODEL_TYPE': 'qwen3_32b',
                   'DENSE_LAYERS': '12', 'DENSE_HIDDEN': '5120', 'DENSE_FFN': '25600', 'DENSE_HEADS': '64',
                   'MBS': '2', 'GBS': '16', 'SEQ_LEN': '4352', 'TRAIN_ITERS': str(iterations),
                   'AUTO_ACTIVATION_MEMORY': '1', 'WANDB_MODE': 'disabled', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'HCCL_DETERMINISTIC': 'True', 'CLOSE_MATMUL_K_SHIFT': '1',
                   'PYTHONHASHSEED': '1234', 'LOG_FILE': str(directory / 'train.log'),
                   'TENSORBOARD_DIR': str(directory / 'tensorboard'),
                   'ADAPTIVE_MEM_ITERATION_JSONL': str(directory / 'iterations'),
                   'MEGATRON_OFFLOAD_TRANSPORT_JSONL': str(directory / 'transport')})
    with socket.socket() as connection:
        connection.bind(('127.0.0.1', 0))
        result['MASTER_PORT'] = str(connection.getsockname()[1])
    return result


def capture_arguments():
    selected = {key: os.environ.get(key) for key in ('AUTO_ACTIVATION_MEMORY', 'WANDB_MODE', 'ASCEND_RT_VISIBLE_DEVICES', 'PYTHONPATH')}
    save(Path(os.environ['SMOKE_CAPTURE_FILE']), {'argv': sys.argv[2:], 'environment': selected})
    return int(os.environ['SMOKE_CAPTURE_EXIT'])


def validate_launcher(output):
    subprocess.run(['bash', '-n', str(LAUNCHER)], check=True)
    checks = []
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        directory = Path(temporary)
        fixture = directory / 'torchrun'
        fixture.write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(__file__) + ' --capture-argv "$@"\n')
        fixture.chmod(0o700)
        for code in (0, 23):
            selected = environment(directory, 80)
            selected.update({'PATH': str(directory) + ':' + selected['PATH'],
                             'SMOKE_CAPTURE_FILE': str(directory / 'arguments.json'), 'SMOKE_CAPTURE_EXIT': str(code)})
            completed = subprocess.run(['bash', 'run_offload.sh'], cwd=MEGATRON, env=selected, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            assert completed.returncode == code, (completed.returncode, code)
            captured = json.loads((directory / 'arguments.json').read_text())
            arguments = captured['argv']
            for flag, expected in (('--nproc_per_node', '4'), ('--num-layers', '12'), ('--hidden-size', '5120'),
                                   ('--ffn-hidden-size', '25600'), ('--seq-length', '4352'),
                                   ('--micro-batch-size', '2'), ('--global-batch-size', '16'),
                                   ('--pipeline-model-parallel-size', '4'), ('--train-iters', '80')):
                assert arguments[arguments.index(flag) + 1] == expected, flag
            assert '--accumulate-allreduce-grads-in-fp32' in arguments and '--use-distributed-optimizer' in arguments
            assert '--sequence-parallel' not in arguments and '--swap-optimizer' not in arguments
            assert captured['environment']['WANDB_MODE'] == 'disabled'
            assert captured['environment']['AUTO_ACTIVATION_MEMORY'] == '1'
            checks.append({'requested_exit': code, 'actual_exit': completed.returncode, 'configuration_verified': True})
    origin = importlib.util.find_spec('mindspeed').origin
    assert Path(origin).resolve() == REPOSITORY / 'mindspeed/__init__.py'
    save(output / 'launcher-validation.json', {'passed': True, 'timestamp': now(), 'checks': checks,
                                               'package_origin': origin, 'launcher_sha256': digest(LAUNCHER)})


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def inspect_case(output, phase, iterations):
    directory = output / phase
    execution = json.loads((directory / 'execution.json').read_text())
    assert execution['exit_code'] == 0 and not execution['safety_stop']
    text = (directory / 'train.log').read_text(errors='replace')
    assert '[after training is done]' in text
    assert not re.search(r'Traceback|OutOfMemoryError|out of memory|\[PressureBench\]', text, re.I)
    expected = {'auto_activation_memory': 'True', 'optimizer': 'adam', 'swap_optimizer': 'False',
                'optimizer_cpu_offload': 'False', 'bf16': 'True', 'accumulate_allreduce_grads_in_fp32': 'True',
                'use_distributed_optimizer': 'True', 'tensor_model_parallel_size': '1',
                'pipeline_model_parallel_size': '4', 'expert_model_parallel_size': '1', 'data_parallel_size': '1',
                'num_layers': '12', 'hidden_size': '5120', 'ffn_hidden_size': '25600', 'seq_length': '4352',
                'micro_batch_size': '2', 'global_batch_size': '16', 'train_iters': str(iterations)}
    for key, value in expected.items():
        matches = re.findall(r'^\s*' + key + r'\s+\.{2,}\s+(.*?)\s*$', text, re.M)
        assert matches and all(item == value for item in matches), (key, matches, value)
    steps = [{'iteration': int(index), 'requested': int(total), 'ms': float(duration), 'loss': float(loss),
              'norm': float(norm), 'skipped': int(skipped), 'nan': int(nan)}
             for index, total, duration, loss, norm, skipped, nan in STEP_PATTERN.findall(text)]
    assert [row['iteration'] for row in steps] == list(range(1, iterations + 1))
    assert all(row['requested'] == iterations and row['skipped'] == row['nan'] == 0
               and all(math.isfinite(row[key]) for key in ('ms', 'loss', 'norm')) for row in steps)
    hits = re.findall(r'\[AutoActivationMemory\]\[CACHE_HIT\] rank=(\d+) signature=([a-f0-9]+)', text)
    misses = re.findall(r'\[AutoActivationMemory\]\[CACHE_MISS\] rank=(\d+) signature=([a-f0-9]+)', text)
    selected = misses if phase == 'cold' else hits
    assert not (hits if phase == 'cold' else misses)
    assert len(selected) == 4 and {int(rank) for rank, signature in selected} == set(range(4))
    ranks = []
    for rank in range(4):
        memory = read_rows(directory / f'iterations.pp{rank}.rank{rank}.jsonl')
        traffic = read_rows(directory / f'transport.rank{rank}.jsonl')
        assert len(memory) == len(traffic) == iterations
        assert [row['iteration_zero_based'] for row in memory] == list(range(iterations))
        assert [row['schedule_iteration'] for row in traffic] == list(range(traffic[0]['schedule_iteration'], traffic[0]['schedule_iteration'] + iterations))
        actions, transferred = set(), 0
        for row, transfer in zip(memory, traffic):
            assert row['rank'] == row['pp_rank'] == transfer['rank'] == rank
            assert not row['memory_pressure'] and row['observed_footprint_peak_bytes'] / 2**20 <= row['memory_limit_mb'] + 1e-6
            assert row['microbatch_liveness']['peak_live'] == 4 - rank
            assert row['microbatch_liveness']['live'] == row['microbatch_liveness']['backward'] == 0
            assert transfer['eligible_activation_accounting_valid']
            for group in transfer['groups'].values():
                assert group['d2h_bytes'] == group['h2d_bytes']
                transferred += group['d2h_bytes']
            for label, count in transfer['module_policy_calls'].items():
                module, action = label.rsplit(':', 1)
                assert count == 8
                actions.add(action)
                if action == 'RECOMPUTE':
                    assert transfer['module_policy_calls'][module + ':REPLAY'] == count
                if row['plan_applied'] and action != 'REPLAY':
                    assert row['plan']['decisions'].get(module, 'KEEP') == action
            if row['plan_applied']:
                assert row['plan']['predicted_peak_mb'] <= row['plan']['memory_limit_mb']
        optimized = [row['iteration_zero_based'] + 1 for row in memory if row['plan_applied']]
        unoptimized = [row['iteration_zero_based'] + 1 for row in memory if not row['plan_applied']]
        assert optimized and optimized[-1] == iterations
        if phase == 'cold':
            assert optimized[0] == 9 and unoptimized[:8] == list(range(1, 9))
            assert any(index >= 64 for index in unoptimized), 'Periodic reprofile was not observed'
        else:
            assert optimized[0] == 1 and not unoptimized
        ranks.append({'rank': rank, 'actions': sorted(actions), 'd2h_gib': transferred / 2**30,
                      'peak_mib': max(row['observed_footprint_peak_bytes'] for row in memory) / 2**20,
                      'minimum_guard_margin_mib': min(row['memory_limit_mb'] - row['observed_footprint_peak_bytes'] / 2**20 for row in memory),
                      'optimized_steps': optimized, 'unoptimized_steps': unoptimized,
                      'first_plan': next(row['plan'] for row in memory if row['plan_applied']), 'last_plan': memory[-1]['plan']})
    assert any(rank['d2h_gib'] > 0 for rank in ranks), 'Native offload was not exercised'
    assert {'KEEP', 'OFFLOAD', 'RECOMPUTE'}.issubset({action for rank in ranks for action in rank['actions']})
    return {'passed': True, 'phase': phase, 'updates': iterations, 'settings': expected, 'steps': steps, 'ranks': ranks,
            'cache_signatures': dict(selected), 'cache_hit': phase == 'warm', 'full_gradient_hashes_collected': False,
            'steady_logged_median_ms': statistics.median(row['ms'] for row in steps[8:]),
            'all_logged_seconds': sum(row['ms'] for row in steps) / 1000}


def run_case(output, phase, iterations):
    directory = output / phase
    directory.mkdir()
    assert not devices(), 'Requested devices are occupied'
    selected = environment(directory, iterations)
    save(directory / 'launch.json', {'timestamp': now(), 'argv': ['bash', 'run_offload.sh'], 'cwd': str(MEGATRON),
                                    'settings': {key: selected[key] for key in ('MODEL_TYPE', 'DENSE_LAYERS', 'MBS', 'GBS', 'SEQ_LEN', 'TRAIN_ITERS', 'AUTO_ACTIVATION_MEMORY', 'WANDB_MODE', 'ASCEND_RT_VISIBLE_DEVICES', 'MASTER_PORT', 'PYTHONPATH')}})
    workers, violations = {}, []
    with (directory / 'shell.stderr.log').open('x') as error_output:
        process = subprocess.Popen(['bash', 'run_offload.sh'], cwd=MEGATRON, env=selected,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=error_output, start_new_session=True)
        initial = process_state(process.pid)
        assert initial is not None
        save(directory / 'process.json', initial)
        started, next_device_check = time.monotonic(), 0
        try:
            while process.poll() is None:
                for rank in range(4):
                    path = directory / f'iterations.pp{rank}.rank{rank}.jsonl'
                    if not path.exists():
                        continue
                    for line in path.read_text().splitlines():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if row['memory_pressure'] or row['observed_footprint_peak_bytes'] / 2**20 > row['memory_limit_mb'] + 1e-6:
                            violations.append({'rank': rank, 'iteration': row['iteration_zero_based'] + 1})
                if violations:
                    raise RuntimeError('Native memory safety guard failed')
                if time.monotonic() - started > max(1800, iterations * 45):
                    raise TimeoutError('Registered training runtime limit exceeded')
                if time.monotonic() >= next_device_check:
                    for device, pid in devices().items():
                        state = process_state(pid)
                        assert state is not None and state['uid'] == os.getuid() and descendant(pid, process.pid), (device, pid, 'Foreign device process')
                        assert b'pretrain_gpt.py' in (Path('/proc') / str(pid) / 'cmdline').read_bytes()
                        state['device'] = device
                        workers[pid] = state
                    save(directory / 'workers.json', list(workers.values()))
                    next_device_check = time.monotonic() + 15
                time.sleep(2)
        finally:
            if process.poll() is None:
                for pid, recorded in workers.items():
                    state = process_state(pid)
                    if state is not None and state['uid'] == os.getuid() and state['start_ticks'] == recorded['start_ticks']:
                        os.kill(pid, signal.SIGTERM)
                state = process_state(process.pid)
                assert state is not None and state['start_ticks'] == initial['start_ticks'] and state['uid'] == os.getuid()
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    state = process_state(process.pid)
                    assert state is not None and state['start_ticks'] == initial['start_ticks']
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            save(directory / 'execution.json', {'exit_code': process.returncode, 'timestamp': now(), 'wall_seconds': time.monotonic() - started,
                                                'safety_stop': bool(violations), 'violations': violations, 'worker_pids': sorted(workers)})
    assert len(workers) == 4 and {record['device'] for record in workers.values()} == set(range(4))
    return inspect_case(output, phase, iterations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    options = parser.parse_args()
    output = options.output.resolve()
    assert output.is_relative_to(PROJECT / 'reports')
    output.mkdir(parents=True, exist_ok=True)
    if options.audit_only:
        assert (output / 'matrix.exit').read_text().strip() == '0'
        evidence = [inspect_case(output, phase, iterations) for phase, iterations in CASES]
        original = json.loads((output / 'summary.json').read_text())
        assert original['cases'] == evidence
        assert all(digest(path) == checksum for path, checksum in original['source_sha256'].items())
        save(output / 'independent-audit.json', {'passed': True, 'timestamp': now(), 'updates': 100,
                                                'full_gradient_hashes_collected': False, 'source_unchanged': True})
        print('Independent native audit passed')
        return
    assert not (output / 'controller.json').exists()
    validate_launcher(output)
    if options.validate_only:
        print('Launcher argument, offline logging and exit-code checks passed; no training started')
        return
    integrated = json.loads((PROJECT / 'reports/activation-memory/evidence.json').read_text())['integrated_runtime_sha256']
    assert all(digest(path) == value for path, value in integrated.items())
    sources = {**integrated, str(LAUNCHER): digest(LAUNCHER), str(Path(__file__).resolve()): digest(__file__),
               str(MEGATRON / 'pretrain_gpt.py'): digest(MEGATRON / 'pretrain_gpt.py'),
               str(MEGATRON / 'megatron/core/distributed/param_and_grad_buffer.py'): digest(MEGATRON / 'megatron/core/distributed/param_and_grad_buffer.py')}
    save(output / 'registration.json', {'timestamp': now(), 'cases': CASES, 'source_sha256': sources,
                                       'original_launcher_used': True, 'source_overlay_used': False,
                                       'physical_devices': [0, 1, 2, 3], 'optimizer_swap': False,
                                       'full_model_32b': False, 'full_gradient_hashes_planned': False})
    save(output / 'controller.json', process_state(os.getpid()))
    status = 1
    try:
        evidence = []
        for phase, iterations in CASES:
            save(output / 'current-case.json', {'phase': phase, 'iterations': iterations})
            print('Starting', phase, iterations, now(), flush=True)
            evidence.append(run_case(output, phase, iterations))
            assert all(digest(path) == checksum for path, checksum in sources.items())
            save(output / 'progress.json', {'completed': [item['phase'] for item in evidence]})
            print('Accepted', phase, now(), flush=True)
        assert evidence[0]['cache_signatures'] == evidence[1]['cache_signatures']
        overlap = list(zip(evidence[0]['steps'], evidence[1]['steps']))
        summary = {'passed': True, 'timestamp': now(), 'cases': evidence, 'source_sha256': sources,
                   'total_updates': 100, 'original_launcher_used': True, 'source_overlay_used': False,
                   'full_gradient_hashes_collected': False, 'performance_ab_claimed': False,
                   'logged_loss_max_abs_difference': max(abs(before['loss'] - after['loss']) for before, after in overlap),
                   'logged_norm_max_relative_difference': max(abs(before['norm'] - after['norm']) / max(abs(before['norm']), 1e-30) for before, after in overlap)}
        save(output / 'summary.json', summary)
        status = 0
    except BaseException as error:
        save(output / 'failure.json', {'timestamp': now(), 'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        (output / 'matrix.exit').write_text(str(status) + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--capture-argv':
        raise SystemExit(capture_arguments())
    main()
