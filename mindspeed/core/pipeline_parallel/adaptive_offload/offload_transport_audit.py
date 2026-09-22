import json
import os
from pathlib import Path

import torch


class OffloadTransportAudit:
    def __init__(self):
        self.path = os.environ.get("MEGATRON_OFFLOAD_TRANSPORT_JSONL")
        self.enabled = bool(self.path)
        self.iteration = -1
        self.groups = {}
        self.policy_calls = {}
        self.pinned_memory_pool = None

    def start(self):
        self.iteration += 1
        self.groups = {}
        self.policy_calls = {}

    def record_policy(self, module, action):
        if self.enabled:
            key = module + ':' + action
            self.policy_calls[key] = self.policy_calls.get(key, 0) + 1

    def record(self, phase, name, layer, last_layer, byte_count):
        if not self.enabled:
            return
        key = f"layer{layer}:{name}"
        record = self.groups.setdefault(key, {
            "name": name, "layer": layer, "last_layer": last_layer,
            "candidate_bytes": 0, "d2h_bytes": 0, "h2d_bytes": 0,
            "candidate_tensors": 0, "d2h_tensors": 0, "h2d_tensors": 0,
            "kept_bytes": 0, "kept_tensors": 0,
            "deduplicated_bytes": 0, "deduplicated_tensors": 0,
        })
        record[f"{phase}_bytes"] += byte_count
        record[f"{phase}_tensors"] += 1

    def finish(self):
        if not self.enabled or (not self.groups and os.environ.get('AUTO_ACTIVATION_MEMORY') != '1'):
            return
        valid = all(
            group["candidate_bytes"] == group["d2h_bytes"] + group["kept_bytes"] + group["deduplicated_bytes"]
            and group["d2h_bytes"] == group["h2d_bytes"]
            and group["candidate_tensors"] == group["d2h_tensors"] + group["kept_tensors"] + group["deduplicated_tensors"]
            and group["d2h_tensors"] == group["h2d_tensors"]
            for group in self.groups.values()
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        result = {
            "schedule_iteration": self.iteration, "rank": rank,
            "eligible_activation_accounting_valid": valid,
            "groups": self.groups,
            "module_policy_calls": self.policy_calls,
            "observed_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
            "observed_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
            "pinned_memory_pool": self.pinned_memory_pool.stats if self.pinned_memory_pool is not None else {},
            "main_thread_cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        }
        if self.path:
            path = Path(f"{self.path}.rank{rank}.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(result, sort_keys=True) + "\n")
        if not valid:
            mismatched = [name for name, group in self.groups.items()
                          if group["candidate_bytes"] != group["d2h_bytes"] + group["kept_bytes"] + group["deduplicated_bytes"]
                          or group["d2h_bytes"] != group["h2d_bytes"]
                          or group["candidate_tensors"] != group["d2h_tensors"] + group["kept_tensors"] + group["deduplicated_tensors"]
                          or group["d2h_tensors"] != group["h2d_tensors"]]
            raise RuntimeError(f"Activation transport accounting failed: {mismatched}")
