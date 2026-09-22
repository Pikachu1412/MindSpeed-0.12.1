import hashlib
import time
import uuid
from pathlib import Path


def _node_identity():
    try:
        identity = uuid.UUID(Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, ValueError):
        return (0, 0)
    encoded = hashlib.sha256(identity.bytes).digest()
    maximum = (1 << 63) - 1
    return tuple(int.from_bytes(encoded[offset:offset + 8], 'big') & maximum for offset in (0, 8))


def calibration_schedule(identities, enabled):
    if not identities or len(identities) != len(enabled):
        raise ValueError('Calibration participants must be a nonempty aligned sequence')
    if any(type(active) is not bool for active in enabled):
        raise ValueError('Calibration eligibility must be boolean')
    if any(not isinstance(identity, (tuple, list)) or len(identity) != 2
           or any(type(number) is not int or number < 0 or number >= 1 << 63 for number in identity)
           for identity in identities):
        raise ValueError('Invalid calibration host identity')
    if any(tuple(identity) == (0, 0) for identity in identities):
        return tuple((rank,) for rank, active in enumerate(enabled) if active)
    groups = {}
    for rank, (identity, active) in enumerate(zip(identities, enabled)):
        if active:
            groups.setdefault(tuple(identity), []).append(rank)
    count = max((len(group) for group in groups.values()), default=0)
    return tuple(tuple(group[phase] for group in groups.values() if phase < len(group))
                 for phase in range(count))


def coordinate_bandwidth_calibration(measure, enabled, torch_module):
    started_ns = time.monotonic_ns()
    distributed = torch_module.distributed.is_initialized()
    rank = torch_module.distributed.get_rank() if distributed else 0
    world_size = torch_module.distributed.get_world_size() if distributed else 1
    result = None
    call_started_ns = None
    call_finished_ns = None
    node = _node_identity()
    if world_size == 1:
        if enabled:
            call_started_ns = time.monotonic_ns()
            result = measure()
            call_finished_ns = time.monotonic_ns()
        phases = ((0,),) if enabled else ()
        identities = [node]
        eligibility = [bool(enabled)]
    else:
        device = 'cuda' if torch_module.cuda.is_available() else 'cpu'
        local = torch_module.tensor([*node, int(enabled)], dtype=torch_module.int64, device=device)
        gathered = [torch_module.empty_like(local) for _ in range(world_size)]
        torch_module.distributed.all_gather(gathered, local)
        rows = [value.cpu().tolist() for value in gathered]
        if any(row[2] not in (0, 1) for row in rows):
            raise ValueError('Invalid gathered calibration eligibility')
        identities = [tuple(row[:2]) for row in rows]
        eligibility = [bool(row[2]) for row in rows]
        phases = calibration_schedule(identities, eligibility)
        failure = None
        if phases:
            torch_module.distributed.barrier()
            for participants in phases:
                try:
                    if rank in participants:
                        call_started_ns = time.monotonic_ns()
                        result = measure()
                        call_finished_ns = time.monotonic_ns()
                except Exception as exception:
                    failure = exception
                    call_finished_ns = time.monotonic_ns()
                finally:
                    torch_module.distributed.barrier()
            failed = torch_module.tensor([int(failure is not None)], dtype=torch_module.int32, device=device)
            torch_module.distributed.all_reduce(failed, op=torch_module.distributed.ReduceOp.MAX)
            if failed.item():
                raise RuntimeError('Coordinated bandwidth calibration callback failed on one or more ranks') from failure
    return result, {
        'rank': rank, 'world_size': world_size, 'enabled': bool(enabled),
        'node_identity': list(node), 'participant_enabled': eligibility,
        'phases': [list(phase) for phase in phases],
        'coordination': ('single_process' if world_size == 1 else
                         'global_fallback' if any(tuple(identity) == (0, 0) for identity in identities)
                         else 'per_kernel_boot_id'),
        'started_ns': started_ns, 'finished_ns': time.monotonic_ns(),
        'call_started_ns': call_started_ns, 'call_finished_ns': call_finished_ns,
    }
