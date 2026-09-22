"""Deterministic joint activation-policy search, independent of torch/device state."""

import itertools
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping, Optional, Tuple


KEEP = "KEEP"
OFFLOAD = "OFFLOAD"
RECOMPUTE = "RECOMPUTE"
BACKWARD_ORDER = (
    "mlp_norm", "moe_act", "expert_fc1", "attn_norm", "attn_proj", "core_attn", "qkv_linear"
)
MODULE_GROUPS = {
    "attn_norm": ("attn_norm",),
    "attention": ("qkv_linear", "core_attn", "attn_proj"),
    "mlp_norm": ("mlp_norm",),
    "mlp": ("expert_fc1", "moe_act"),
}


@dataclass(frozen=True)
class GroupProfile:
    activation_mb: float
    forward_ms: float
    backward_ms: float
    d2h_ms: float
    h2d_ms: float
    released_mb: float = 0.0
    resident_mb: float = 0.0
    keep_mb: Optional[float] = None
    d2h_storage_ratio: float = 1.0

    @property
    def keep_activation_mb(self):
        return self.activation_mb if self.keep_mb is None else self.keep_mb


@dataclass(frozen=True)
class ModuleProfile:
    recompute_ms: float
    input_mb: float
    workspace_mb: float
    measured: bool = True


@dataclass(frozen=True)
class ChildRecomputeProfile:
    """Measured alternative to retaining one entire parent activation group.

    retained_mb is the storage union for the whole parent under this alternative,
    including checkpoint inputs and non-offloadable storage exactly once. It is
    not a saving to subtract from transfer bytes. workspace_mb bounds additional
    reconstruction allocations, including both temporary and restored outputs.
    Child KEEP means no independent child checkpoint; the parent still owns its
    normal KEEP, OFFLOAD or full-module RECOMPUTE operation.
    """

    parent_group: str
    recompute_ms: float
    retained_mb: float
    workspace_mb: float
    measured: bool = True


@dataclass(frozen=True)
class MemoryBudget:
    baseline_peak_mb: float
    capacity_mb: float
    reserve_mb: float
    extra_budget_mb: Optional[float] = None
    activation_margin: float = 1.1
    allocator_unavailable_mb: float = 0.0
    activation_baseline_peak_mb: Optional[float] = None

    @property
    def limit_mb(self) -> float:
        physical_limit = self.capacity_mb - self.reserve_mb - self.allocator_unavailable_mb
        if self.extra_budget_mb is not None:
            return min(physical_limit, self.baseline_peak_mb + self.extra_budget_mb)
        return physical_limit


@dataclass(frozen=True)
class ScheduleConfig:
    layers: Tuple[Tuple[str, ...], ...]
    inflight_microbatches: int
    adjacent_prefetch: int = 1
    cross_layer_prefetch: bool = True
    last_layer_keep: bool = False
    recompute_factor: float = 1.25
    transfer_factor: float = 1.1
    serialized_recompute: bool = False
    prefetch_memory_mb: float = 0.0
    baseline_recompute: bool = False
    synchronous_d2h: bool = False
    immediate_backward_keep: bool = False
    unified_transport: bool = False
    d2h_slots: int = 1
    host_module_ms: float = 0.0
    host_offload_ms: float = 0.0
    offload_only_callbacks: bool = False


@dataclass
class StrategyPlan:
    decisions: Dict[str, str]
    predicted_overhead_ms: float
    predicted_peak_mb: float
    memory_limit_mb: float
    recompute_ms: float
    reload_stall_ms: float
    offload_tail_ms: float
    candidates: int = 0
    feasible_candidates: int = 0
    transfer_work_ms: float = 0.0
    transfer_count: int = 0
    search_complete: bool = True
    execution: Dict = field(default_factory=dict)
    host_overhead_ms: float = 0.0

    @property
    def keep_groups(self):
        return {name for name, action in self.decisions.items() if action == KEEP}

    @property
    def recompute_groups(self):
        return {name for name, action in self.decisions.items() if action == RECOMPUTE}

    def to_dict(self):
        return asdict(self)


class NoFeasiblePlanError(RuntimeError):
    pass


class UnifiedOffloadOptimizer:
    """Minimize modeled overhead with exact or explicitly bounded search.

    KEEP/OFFLOAD are independent per group. RECOMPUTE is atomic over each actual
    checkpoint call. Missing checkpoint measurements exclude that action rather
    than substituting a sub-group's forward time. All predictions are conservative
    estimates, not a guarantee for unseen shapes, allocator or external workloads.
    """

    def __init__(
        self,
        groups: Mapping[str, GroupProfile],
        modules: Mapping[str, ModuleProfile],
        memory: MemoryBudget,
        schedule: ScheduleConfig,
        module_groups=None,
        nested_groups=None,
        max_exact_candidates=10000,
        child_recomputes=None,
    ):
        self.groups = dict(groups)
        self.modules = dict(modules)
        self.memory = memory
        self.schedule = schedule
        self.module_groups = dict(MODULE_GROUPS if module_groups is None else module_groups)
        self.instance_mode = module_groups is not None
        self.nested_groups = nested_groups if nested_groups is not None else {
            'attn_norm': MODULE_GROUPS['attention'], 'mlp_norm': MODULE_GROUPS['mlp']}
        self.max_exact_candidates = max_exact_candidates
        self.child_recomputes = dict(child_recomputes or {})
        self._validate()
        self.child_recomputes = {name: profile for name, profile in self.child_recomputes.items()
                                 if profile.measured and profile.recompute_ms > 0}
        self.parent_children = {}
        for name, profile in self.child_recomputes.items():
            self.parent_children.setdefault(profile.parent_group, []).append(name)
        self.group_modules = {
            group: module for module, members in self.module_groups.items() for group in members
        }
        self.backward = [
            (layer_index, group)
            for layer_index in reversed(range(len(schedule.layers)))
            for group in (schedule.layers[layer_index] if self.instance_mode else BACKWARD_ORDER)
            if group in schedule.layers[layer_index]
        ]

    def _validate(self):
        known = {group for members in self.module_groups.values() for group in members}
        unknown = set(self.groups) - known
        if unknown:
            raise ValueError(f"Unsupported offload groups: {sorted(unknown)}")
        if any(sum(group in members for members in self.module_groups.values()) != 1 for group in self.groups):
            raise ValueError('Every active group must belong to exactly one checkpoint boundary')
        for name, profile in self.child_recomputes.items():
            if not isinstance(name, str) or not name or name in self.groups:
                raise ValueError('Child checkpoint names must be distinct from parent groups')
            if profile.parent_group not in self.groups or not self.instance_mode:
                raise ValueError('Child checkpoint requires an existing layer-instance parent')
            owners = [tuple(group for group in members if group in self.groups)
                      for members in self.module_groups.values() if profile.parent_group in members]
            if owners != [(profile.parent_group,)]:
                raise ValueError('Child checkpoint requires a single-group parent boundary')
            if sum(layer.count(profile.parent_group) for layer in self.schedule.layers) != 1:
                raise ValueError('Child checkpoint parent must occur exactly once in the schedule')
            values = (profile.recompute_ms, profile.retained_mb, profile.workspace_mb)
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError('Child checkpoint measurements must be finite and nonnegative')
            if not isinstance(profile.measured, bool):
                raise ValueError('Child checkpoint measurement validity must be boolean')
        if not isinstance(self.max_exact_candidates, int) or self.max_exact_candidates < 1:
            raise ValueError('Exact-search candidate budget must be positive')
        if not math.isfinite(self.schedule.prefetch_memory_mb) or self.schedule.prefetch_memory_mb < 0:
            raise ValueError('Prefetch memory must be finite and nonnegative')
        for profile in self.groups.values():
            values = asdict(profile)
            if profile.keep_mb is None:
                values.pop("keep_mb")
            if any(not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError("Group measurements must be finite and nonnegative")
            if profile.d2h_storage_ratio < 1.0:
                raise ValueError("D2H storage ratio must preserve the original staging bound")
        for profile in self.modules.values():
            values = (profile.recompute_ms, profile.input_mb, profile.workspace_mb)
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("Module measurements must be finite and nonnegative")
        values = (self.memory.baseline_peak_mb, self.memory.capacity_mb,
                  self.memory.reserve_mb, self.memory.allocator_unavailable_mb)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Memory telemetry must be finite and nonnegative")
        if self.memory.activation_baseline_peak_mb is not None and (
            not math.isfinite(self.memory.activation_baseline_peak_mb) or self.memory.activation_baseline_peak_mb < 0
        ):
            raise ValueError('Activation-phase baseline must be finite and nonnegative')
        if self.memory.extra_budget_mb is not None and (
            not math.isfinite(self.memory.extra_budget_mb) or self.memory.extra_budget_mb < 0
        ):
            raise ValueError("Additional memory budget must be finite and nonnegative")
        factors = (
            self.memory.activation_margin,
            self.schedule.recompute_factor,
            self.schedule.transfer_factor,
        )
        if any(not math.isfinite(value) or value < 1 for value in factors):
            raise ValueError("Prediction safety factors must be finite and at least 1")
        if self.schedule.inflight_microbatches < 1 or self.schedule.adjacent_prefetch < 0 or self.schedule.d2h_slots < 1:
            raise ValueError("Invalid pipeline concurrency or prefetch depth")
        if any(not math.isfinite(value) or value < 0 for value in (
            self.schedule.host_module_ms, self.schedule.host_offload_ms
        )):
            raise ValueError('Host overhead costs must be finite and nonnegative')
        if not self.schedule.layers or any(
            set(layer) - self.groups.keys() or len(set(layer)) != len(layer)
            for layer in self.schedule.layers
        ):
            raise ValueError("Layer layout must contain known, nonduplicated groups")

    def _candidate_choices(self):
        choices = []
        for module, members in self.module_groups.items():
            active = tuple(group for group in members if group in self.groups)
            if not active:
                continue
            options = [dict(zip(active, actions)) for actions in itertools.product((KEEP, OFFLOAD), repeat=len(active))]
            profile = self.modules.get(module)
            if profile is not None and profile.measured and profile.recompute_ms > 0:
                options.append(dict.fromkeys(active, RECOMPUTE))
            children = [name for group in active for name in self.parent_children.get(group, ())]
            if children:
                defaults = dict.fromkeys(children, KEEP)
                options = [dict(option, **defaults) for option in options]
                for name in children:
                    option = dict.fromkeys(active, KEEP)
                    option.update(defaults)
                    option[name] = RECOMPUTE
                    options.append(option)
            choices.append(options)
        return choices

    def candidate_decisions(self):
        for combination in itertools.product(*self._candidate_choices()):
            yield {group: action for unit in combination for group, action in unit.items()}

    def _selected_children(self, decisions):
        return {profile.parent_group: profile for name, profile in self.child_recomputes.items()
                if decisions[name] == RECOMPUTE}

    def _is_offloaded(self, layer, group, decisions):
        return decisions[group] == OFFLOAD and not (
            self.schedule.last_layer_keep and layer == len(self.schedule.layers) - 1
        )

    def _exclusive_compute(self, group, direction):
        value = getattr(self.groups[group], direction)
        children = self.nested_groups.get(group, ())
        if not children:
            return max(0.0, value)
        return max(0.0, value - sum(
            getattr(self.groups[child], direction) for child in children if child in self.groups
        ))

    def _memory_peak(self, decisions):
        children = self._selected_children(decisions)
        child_workspaces = [profile.workspace_mb for profile in children.values()]
        child_workspace = (max(child_workspaces, default=0.0) if self.schedule.serialized_recompute
                           else sum(child_workspaces))
        if self.schedule.baseline_recompute:
            retained = sum(children[group].retained_mb if group in children else
                           self.groups[group].keep_activation_mb + self.groups[group].resident_mb
                           for layer in self.schedule.layers for group in layer
                           if decisions[group] == KEEP) * self.schedule.inflight_microbatches
            retained += sum(profile.resident_mb for group, profile in self.groups.items()
                            if decisions[group] == OFFLOAD) * self.schedule.inflight_microbatches
            immediate = sum(max(self.groups[group].activation_mb, self.groups[group].keep_activation_mb) for group in self.schedule.layers[-1]
                            if decisions[group] == OFFLOAD) if self.schedule.immediate_backward_keep else 0.0
            staging = max((self.groups[group].activation_mb for group, action in decisions.items()
                           if action == OFFLOAD), default=0.0)
            storage_ratio = max((self.groups[group].d2h_storage_ratio for group, action in decisions.items()
                                 if action == OFFLOAD), default=1.0)
            prefetch = self.schedule.prefetch_memory_mb
            if self.schedule.unified_transport:
                prefetch = staging * (self.schedule.adjacent_prefetch + 1)
                staging *= self.schedule.d2h_slots
            staging *= storage_ratio
            phase = self.memory.activation_baseline_peak_mb
            phase = self.memory.baseline_peak_mb if phase is None else phase
            return max(self.memory.baseline_peak_mb, phase + self.memory.activation_margin * (
                retained + staging + immediate + prefetch + child_workspace))
        retained_mb = 0.0
        workspaces = list(child_workspaces)
        for layer in self.schedule.layers:
            retained_mb += sum(
                children[group].retained_mb if group in children else self.groups[group].keep_activation_mb
                for group in layer if decisions[group] == KEEP
            )
            recomputed = {self.group_modules[group] for group in layer if decisions[group] == RECOMPUTE}
            for module in recomputed:
                profile = self.modules[module]
                retained_mb += profile.input_mb
                workspaces.append(profile.workspace_mb)
        extra_mb = (
            retained_mb * self.schedule.inflight_microbatches
            + (max(workspaces, default=0.0) if self.schedule.serialized_recompute else sum(workspaces))
            + self.schedule.prefetch_memory_mb
            + sum(
                sum(self.groups[group].released_mb for group in members
                    if group in decisions and decisions[group] == OFFLOAD)
                for members in self.module_groups.values()
                if any(decisions.get(group) == KEEP for group in members)
            )
        ) * self.memory.activation_margin
        return self.memory.baseline_peak_mb + extra_mb

    def evaluate(self, decisions: Mapping[str, str]) -> StrategyPlan:
        return self._evaluate_with_memory_peak(decisions)

    def _evaluate_with_memory_peak(self, decisions: Mapping[str, str], predicted_peak_mb: Optional[float] = None) -> StrategyPlan:
        if set(decisions) != set(self.groups) | set(self.child_recomputes) or any(
            action not in (KEEP, OFFLOAD, RECOMPUTE) for action in decisions.values()
        ):
            raise ValueError("A plan must assign exactly one action to every active group")
        for name, profile in self.child_recomputes.items():
            if decisions[name] == OFFLOAD or (decisions[name] == RECOMPUTE
                                             and decisions[profile.parent_group] != KEEP):
                raise ValueError('Child recompute requires parent KEEP and cannot overlap parent offload/recompute')
        for parent, names in self.parent_children.items():
            if sum(decisions[name] == RECOMPUTE for name in names) > 1:
                raise ValueError('Overlapping child alternatives require a joint storage measurement')
        children = self._selected_children(decisions)
        for module, members in self.module_groups.items():
            actions = [decisions[group] for group in members if group in decisions]
            if RECOMPUTE in actions and (
                any(action != RECOMPUTE for action in actions)
                or module not in self.modules
                or not self.modules[module].measured
                or self.modules[module].recompute_ms <= 0
            ):
                raise ValueError("Recompute must cover an entire measured checkpoint module")

        ready = {}
        h2d_clock = 0.0
        compute_clock = 0.0
        stall_total = 0.0
        recompute_total = 0.0
        recomputed = set()
        prefetched_layers = set()
        positions = {item: index for index, item in enumerate(self.backward)}
        offloaded = {item for item in self.backward if self._is_offloaded(*item, decisions)}

        def issue(item, release_time):
            nonlocal h2d_clock
            layer, group = item
            if item in ready or item not in offloaded:
                return
            h2d_clock = max(h2d_clock, release_time) + self.groups[group].h2d_ms * self.schedule.transfer_factor
            ready[item] = h2d_clock

        for index, item in enumerate(self.backward):
            layer, group = item
            if item in offloaded:
                issue(item, compute_clock)
                stall = max(0.0, ready[item] - compute_clock)
                stall_total += stall
                compute_clock += stall

            has_marker = (self.schedule.unified_transport and not self.schedule.offload_only_callbacks) or decisions[group] == OFFLOAD
            if has_marker and self.schedule.last_layer_keep and layer > 0 and layer not in prefetched_layers:
                for target in self.backward:
                    if target[0] == layer - 1:
                        issue(target, compute_clock)
                prefetched_layers.add(layer)
            elif has_marker and self.schedule.adjacent_prefetch:
                targets = [target for target in self.backward[index + 1:]
                           if target not in ready and target in offloaded]
                for target in targets[:self.schedule.adjacent_prefetch]:
                    issue(target, compute_clock)

            module = self.group_modules[group]
            if decisions[group] == RECOMPUTE and (layer, module) not in recomputed:
                duration = self.modules[module].recompute_ms * self.schedule.recompute_factor
                recompute_total += duration
                compute_clock += duration
                recomputed.add((layer, module))
            if group in children:
                duration = children[group].recompute_ms * self.schedule.recompute_factor
                recompute_total += duration
                compute_clock += duration
            compute_clock += self._exclusive_compute(group, "backward_ms")
            if has_marker and self.schedule.cross_layer_prefetch:
                target = (layer - 1, group)
                if target in positions:
                    issue(target, compute_clock)

        forward_clock = 0.0
        d2h_clock = 0.0
        d2h_pending = []
        d2h_wait = 0.0
        source_limit = self.schedule.d2h_slots * max((self.groups[group].activation_mb
                       for group, action in decisions.items() if action == OFFLOAD), default=0.0)
        transfer_work = 0.0
        transfer_count = 0
        for layer, group in reversed(self.backward):
            if self.schedule.unified_transport and (layer, group) in offloaded:
                d2h_pending = [entry for entry in d2h_pending if entry[0] > forward_clock]
                while d2h_pending and sum(entry[1] for entry in d2h_pending) + self.groups[group].activation_mb > source_limit + 1e-9:
                    completed, _ = d2h_pending.pop(0)
                    d2h_wait += max(0.0, completed - forward_clock)
                    forward_clock = max(forward_clock, completed)
                    d2h_pending = [entry for entry in d2h_pending if entry[0] > forward_clock]
            forward_clock += self._exclusive_compute(group, "forward_ms")
            if (layer, group) in offloaded:
                d2h_clock = max(d2h_clock, forward_clock) + self.groups[group].d2h_ms * self.schedule.transfer_factor
                d2h_pending.append((d2h_clock, self.groups[group].activation_mb))
                transfer_work += (self.groups[group].d2h_ms + self.groups[group].h2d_ms) * self.schedule.transfer_factor
                transfer_count += 2
        offload_tail = (sum(self.groups[group].d2h_ms * self.schedule.transfer_factor
                           for layer, group in self.backward if (layer, group) in offloaded)
                        if self.schedule.synchronous_d2h else d2h_wait + max(0.0, d2h_clock - forward_clock))
        host_overhead = 0.0
        if transfer_count:
            callbacks = 0 if self.schedule.synchronous_d2h else len(self.backward)
            host_overhead = callbacks * self.schedule.host_module_ms + transfer_count / 2 * self.schedule.host_offload_ms
        return StrategyPlan(
            decisions=dict(decisions),
            predicted_overhead_ms=stall_total + recompute_total + offload_tail + host_overhead,
            predicted_peak_mb=self._memory_peak(decisions) if predicted_peak_mb is None else predicted_peak_mb,
            memory_limit_mb=self.memory.limit_mb,
            recompute_ms=recompute_total,
            reload_stall_ms=stall_total,
            offload_tail_ms=offload_tail,
            transfer_work_ms=transfer_work,
            transfer_count=transfer_count,
            host_overhead_ms=host_overhead,
            execution=({'checkpoint_children': {name: profile.parent_group
                        for name, profile in self.child_recomputes.items()}}
                       if self.child_recomputes else {}),
        )

    def _recompute_lower_bound_terms(self):
        terms = []
        visited = set()
        for layer, group in self.backward:
            module = self.group_modules[group]
            if (layer, module) not in visited:
                profile = self.modules.get(module)
                if profile is not None and profile.measured and profile.recompute_ms > 0:
                    terms.append((group, profile.recompute_ms * self.schedule.recompute_factor))
                visited.add((layer, module))
            for child in self.parent_children.get(group, ()):
                terms.append((child, self.child_recomputes[child].recompute_ms * self.schedule.recompute_factor))
        return terms

    def solve(self) -> StrategyPlan:
        choices = self._candidate_choices()
        if math.prod(len(options) for options in choices) > self.max_exact_candidates:
            return self._solve_bounded(choices)
        best = None
        candidates = 0
        feasible = 0
        minimum_peak = math.inf
        memory_limit = self.memory.limit_mb
        unique_zero_cost = not self.schedule.last_layer_keep and {group for layer_index, group in self.backward} == set(self.groups)
        minimum_cost_proven = False
        recompute_terms = self._recompute_lower_bound_terms()
        for decisions in self.candidate_decisions():
            candidates += 1
            peak = self._memory_peak(decisions)
            minimum_peak = min(minimum_peak, peak)
            if peak > memory_limit:
                continue
            feasible += 1
            if minimum_cost_proven:
                continue
            if best is not None and recompute_terms:
                lower_bound = sum((cost for name, cost in recompute_terms if decisions[name] == RECOMPUTE), 0.0)
                if lower_bound > best[0][0]:
                    continue
            candidate = self._evaluate_with_memory_peak(decisions, peak)
            key = (candidate.predicted_overhead_ms, candidate.transfer_work_ms,
                   candidate.transfer_count, candidate.predicted_peak_mb,
                   tuple(candidate.decisions[name] for name in sorted(candidate.decisions)))
            if best is None or key < best[0]:
                best = key, candidate
                minimum_cost_proven = (unique_zero_cost and candidate.predicted_overhead_ms == 0
                                       and candidate.transfer_work_ms == 0 and candidate.transfer_count == 0
                                       and all(action == KEEP for action in candidate.decisions.values()))
        if best is None:
            raise NoFeasiblePlanError(
                f"No safe activation plan: minimum predicted peak {minimum_peak:.1f} MB "
                f"> limit {self.memory.limit_mb:.1f} MB. Reduce microbatch/sequence length, "
                "increase parallelism, or free device memory; no unsafe plan was applied."
            )
        plan = best[1]
        plan.candidates = candidates
        plan.feasible_candidates = feasible
        return plan

    def _solve_bounded(self, choices):
        evaluated = {}
        minimum_peak = math.inf
        memory_limit = self.memory.limit_mb
        search_limit = None

        def score(decisions):
            nonlocal minimum_peak
            signature = tuple(sorted(decisions.items()))
            if signature not in evaluated:
                if search_limit is not None and len(evaluated) >= search_limit:
                    return None
                peak = self._memory_peak(decisions)
                minimum_peak = min(minimum_peak, peak)
                evaluated[signature] = None if peak > memory_limit else self._evaluate_with_memory_peak(decisions, peak)
            plan = evaluated[signature]
            if plan is None:
                return None
            return (plan.predicted_overhead_ms, plan.transfer_work_ms, plan.transfer_count,
                    plan.predicted_peak_mb, signature)

        def descend(seed, order):
            key = score(seed)
            if key is None:
                return None
            current = dict(seed)
            for _ in range(4):
                improved = False
                for options in order:
                    selected = current
                    selected_key = key
                    for option in options:
                        proposed = dict(current, **option)
                        candidate_key = score(proposed)
                        if candidate_key is not None and candidate_key < selected_key:
                            selected, selected_key = proposed, candidate_key
                    if selected_key < key:
                        current, key, improved = selected, selected_key, True
                if not improved:
                    break
            return key, evaluated[tuple(sorted(current.items()))]

        offload_seed = dict.fromkeys(self.groups, OFFLOAD)
        offload_seed.update(dict.fromkeys(self.child_recomputes, KEEP))
        seeds = [offload_seed]
        for preferred in (KEEP, RECOMPUTE):
            seed = {}
            for options in choices:
                option = next((option for option in options if all(action == preferred for group, action
                               in option.items() if group in self.groups)), options[0])
                seed.update(option)
            seeds.append(seed)
        best = None
        for seed in seeds:
            candidate = descend(seed, choices)
            if candidate is not None and (best is None or candidate[0] < best[0]):
                best = candidate
        if best is None:
            raise NoFeasiblePlanError(f'No safe activation plan in bounded search: tested minimum {minimum_peak:.1f} MB, limit {self.memory.limit_mb:.1f} MB; no unsafe policy applied')
        search_limit = max(len(evaluated), self.max_exact_candidates)
        minimum_cost_proven = (not self.schedule.last_layer_keep
                               and {group for layer_index, group in self.backward} == set(self.groups)
                               and best[1].predicted_overhead_ms == 0
                               and best[1].transfer_work_ms == 0 and best[1].transfer_count == 0
                               and all(action == KEEP for action in best[1].decisions.values()))
        if not minimum_cost_proven:
            reverse_order = tuple(reversed(choices))
            for seed in seeds:
                candidate = descend(seed, reverse_order)
                if candidate is not None and candidate[0] < best[0]:
                    best = candidate
            current = best[1].decisions
            exhausted = False
            for first, second in itertools.combinations(choices, 2):
                for first_option, second_option in itertools.product(first, second):
                    proposed = dict(current, **first_option, **second_option)
                    signature = tuple(sorted(proposed.items()))
                    if signature not in evaluated and len(evaluated) >= search_limit:
                        exhausted = True
                        break
                    candidate_key = score(proposed)
                    if candidate_key is not None and candidate_key < best[0]:
                        best = candidate_key, evaluated[signature]
                if exhausted:
                    break
        plan = best[1]
        plan.candidates = len(evaluated)
        plan.feasible_candidates = sum(candidate is not None for candidate in evaluated.values())
        plan.search_complete = False
        return plan
