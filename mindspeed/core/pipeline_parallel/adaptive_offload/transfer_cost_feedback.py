import math
import time
from dataclasses import replace

from .unified_offload_optimizer import KEEP, OFFLOAD, RECOMPUTE, UnifiedOffloadOptimizer


class TransferCostFeedback:
    def __init__(self, models, selected_model, plan, statistics, iteration, solve_time_ms):
        if not math.isfinite(solve_time_ms) or solve_time_ms <= 0:
            raise ValueError('Transfer feedback requires a measured positive solve cost')
        self.models = tuple(models)
        self.selected_model = selected_model
        self.decisions = dict(plan.decisions)
        self.solve_time_ms = solve_time_ms
        self.reference_counts = {
            name: (statistics[name].d2h_sample_count, statistics[name].h2d_sample_count)
            for name, action in self.decisions.items() if action == OFFLOAD and name in selected_model.groups}
        self.last_signature = tuple(sorted(self.reference_counts.items()))
        self.last_window_iteration = iteration
        self.window_count = 0
        self.last_screen_cost = plan.predicted_overhead_ms
        self.requested = False
        self.last_probe = {}

    def _updated_groups(self, statistics):
        groups = dict(self.selected_model.groups)
        counts = {}
        previous_counts = dict(self.last_signature)
        for name, reference in self.reference_counts.items():
            stats = statistics.get(name)
            if stats is None:
                return None
            current = (stats.d2h_sample_count, stats.h2d_sample_count)
            if any(type(count) is not int or count < previous
                   for count, previous in zip(current, previous_counts[name])):
                return None
            counts[name] = current
            if any(count <= previous for count, previous in zip(current, reference)):
                return None
            values = (stats.d2h_time_ms, stats.h2d_time_ms)
            if any(not math.isfinite(value) or value < 0 for value in values):
                return None
            groups[name] = replace(groups[name], d2h_ms=max(values[0], 0.001), h2d_ms=max(values[1], 0.001))
        return groups, tuple(sorted(counts.items()))

    @staticmethod
    def _repriced_model(model, groups):
        return UnifiedOffloadOptimizer(
            groups, model.modules, model.memory, model.schedule,
            module_groups=model.module_groups, nested_groups=model.nested_groups,
            max_exact_candidates=model.max_exact_candidates, child_recomputes=model.child_recomputes)

    def ready(self, statistics, iteration, remaining_iterations):
        if (self.requested or not self.reference_counts or remaining_iterations <= 0
                or iteration <= self.last_window_iteration):
            return False
        updated = self._updated_groups(statistics)
        if updated is None:
            return False
        groups, signature = updated
        if signature == self.last_signature:
            return False
        self.last_signature = signature
        if iteration != self.last_window_iteration:
            self.window_count += 1
            self.last_window_iteration = iteration
        if self.window_count < 2:
            return False
        started = time.perf_counter_ns()
        held_model = self._repriced_model(self.selected_model, groups)
        held = held_model.evaluate(self.decisions)
        if held.predicted_peak_mb > held_model.memory.limit_mb:
            return False
        drift = abs(held.predicted_overhead_ms - self.last_screen_cost)
        if drift < max(1.0, abs(self.last_screen_cost) * 0.02):
            self.last_probe = {'reason': 'exposed_cost_stable', 'iteration': iteration,
                               'window_count': self.window_count, 'drift_ms': drift,
                               'probe_time_ms': (time.perf_counter_ns() - started) / 1e6}
            return False
        self.last_screen_cost = held.predicted_overhead_ms
        candidates = [self.decisions]
        for name in self.reference_counts:
            for action in (KEEP, RECOMPUTE):
                candidate = dict(self.decisions)
                candidate[name] = action
                candidates.append(candidate)
        candidates.append({name: RECOMPUTE if name in self.reference_counts else action
                           for name, action in self.decisions.items()})
        unique = {tuple(sorted(candidate.items())): candidate for candidate in candidates}
        best = held
        evaluated = 0
        for original in self.models:
            model = held_model if original is self.selected_model else self._repriced_model(original, groups)
            for candidate in unique.values():
                try:
                    value = model.evaluate(candidate)
                except ValueError:
                    continue
                evaluated += 1
                if value.predicted_peak_mb <= model.memory.limit_mb and value.predicted_overhead_ms < best.predicted_overhead_ms:
                    best = value
        probe_ms = (time.perf_counter_ns() - started) / 1e6
        horizon = min(32, remaining_iterations)
        gain = held.predicted_overhead_ms - best.predicted_overhead_ms
        amortized_gain = gain * horizon
        required = 2.0 * (self.solve_time_ms + probe_ms)
        ready = gain >= max(1.0, abs(held.predicted_overhead_ms) * 0.02) and amortized_gain > required
        self.last_probe = {'reason': 'amortized_witness' if ready else 'insufficient_amortized_gain',
                           'iteration': iteration, 'window_count': self.window_count,
                           'held_overhead_ms': held.predicted_overhead_ms,
                           'witness_overhead_ms': best.predicted_overhead_ms,
                           'witness_decisions': dict(best.decisions), 'estimated_gain_ms': gain,
                           'horizon_iterations': horizon, 'amortized_gain_ms': amortized_gain,
                           'required_amortized_gain_ms': required, 'solve_time_ms': self.solve_time_ms,
                           'probe_time_ms': probe_ms, 'evaluated_candidates': evaluated, 'requested': ready}
        self.requested = ready
        return ready
