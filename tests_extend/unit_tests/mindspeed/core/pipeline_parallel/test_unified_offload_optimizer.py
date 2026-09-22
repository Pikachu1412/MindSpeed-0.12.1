import itertools
import math
import random
import unittest
from dataclasses import replace

from mindspeed.core.pipeline_parallel.adaptive_offload.unified_offload_optimizer import (
    BACKWARD_ORDER,
    KEEP,
    MODULE_GROUPS,
    OFFLOAD,
    RECOMPUTE,
    GroupProfile,
    MemoryBudget,
    ModuleProfile,
    NoFeasiblePlanError,
    ScheduleConfig,
    UnifiedOffloadOptimizer,
)


class TestUnifiedOffloadOptimizer(unittest.TestCase):
    def solver(self, groups=None, modules=None, budget=1000, inflight=1, layers=1, **options):
        groups = groups or {"attn_norm": GroupProfile(100, 2, 4, 8, 8)}
        modules = modules if modules is not None else {"attn_norm": ModuleProfile(1, 5, 20)}
        schedule = ScheduleConfig(
            layers=tuple(tuple(groups) for _ in range(layers)),
            inflight_microbatches=inflight,
            recompute_factor=1,
            transfer_factor=1,
            **options,
        )
        return UnifiedOffloadOptimizer(groups, modules, MemoryBudget(100, budget, 0, activation_margin=1), schedule)

    def test_keep_wins_when_activations_fit(self):
        plan = self.solver().solve()
        self.assertEqual(plan.decisions, {"attn_norm": KEEP})
        self.assertEqual(plan.predicted_overhead_ms, 0)

    def test_hidden_transfer_tie_prefers_keep_when_memory_fits(self):
        solver = self.solver(
            groups={"attn_norm": GroupProfile(20, 2, 20, 1, 1)},
            modules={}, layers=2, adjacent_prefetch=0,
            cross_layer_prefetch=False, last_layer_keep=True,
        )
        offload = solver.evaluate({"attn_norm": OFFLOAD})
        self.assertEqual(offload.predicted_overhead_ms, 0)
        self.assertEqual(offload.transfer_work_ms, 2)
        self.assertEqual(offload.transfer_count, 2)
        selected = solver.solve()
        self.assertEqual(selected.decisions, {"attn_norm": KEEP})
        self.assertEqual(selected.transfer_work_ms, 0)
        self.assertEqual(selected.transfer_count, 0)

    def test_hidden_transfer_tie_still_respects_memory_limit(self):
        solver = self.solver(
            groups={"attn_norm": GroupProfile(20, 2, 20, 1, 1)},
            modules={}, budget=120, layers=2, adjacent_prefetch=0,
            cross_layer_prefetch=False, last_layer_keep=True,
        )
        selected = solver.solve()
        self.assertEqual(selected.decisions, {"attn_norm": OFFLOAD})
        self.assertLessEqual(selected.predicted_peak_mb, selected.memory_limit_mb)

    def test_zero_duration_transfer_tie_prefers_no_dispatch(self):
        solver = self.solver(groups={"attn_norm": GroupProfile(20, 2, 20, 0, 0)}, modules={})
        self.assertEqual(solver.solve().decisions, {"attn_norm": KEEP})

    def test_zero_additional_budget_is_not_unlimited_physical_memory(self):
        solver = self.solver(budget=150)
        solver.memory = replace(solver.memory, extra_budget_mb=0)
        self.assertEqual(solver.solve().decisions, {"attn_norm": OFFLOAD})

    def test_nonreleasable_allocator_bytes_reduce_memory_limit(self):
        solver = self.solver(budget=250, modules={})
        solver.memory = replace(solver.memory, allocator_unavailable_mb=80)
        selected = solver.solve()
        self.assertEqual(selected.memory_limit_mb, 170)
        self.assertEqual(selected.decisions, {"attn_norm": OFFLOAD})

    def test_invalid_allocator_telemetry_is_rejected(self):
        solver = self.solver()
        for value in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                UnifiedOffloadOptimizer(solver.groups, solver.modules,
                                        replace(solver.memory, allocator_unavailable_mb=value),
                                        solver.schedule)

    def test_recompute_wins_under_memory_pressure(self):
        plan = self.solver(budget=150).solve()
        self.assertEqual(plan.decisions, {"attn_norm": RECOMPUTE})
        self.assertEqual(plan.predicted_peak_mb, 125)

    def test_expensive_recompute_loses_to_offload(self):
        plan = self.solver(budget=150, modules={"attn_norm": ModuleProfile(50, 5, 20)}).solve()
        self.assertEqual(plan.decisions, {"attn_norm": OFFLOAD})

    def test_missing_module_measurements_exclude_recompute(self):
        plan = self.solver(budget=150, modules={}).solve()
        self.assertEqual(plan.decisions, {"attn_norm": OFFLOAD})

    def test_unmeasured_module_is_not_recomputed(self):
        plan = self.solver(budget=150, modules={"attn_norm": ModuleProfile(0.01, 0, 0, False)}).solve()
        self.assertEqual(plan.decisions, {"attn_norm": OFFLOAD})

    def test_inflight_and_layers_count_together(self):
        solver = self.solver(budget=10000, inflight=8, layers=2)
        keep = solver.evaluate({"attn_norm": KEEP})
        self.assertEqual(keep.predicted_peak_mb, 1700)
        solver.memory = replace(solver.memory, capacity_mb=500)
        self.assertNotEqual(solver.solve().decisions["attn_norm"], KEEP)

    def test_checkpoint_inputs_and_workspace_are_not_free(self):
        solver = self.solver(inflight=4, layers=2)
        plan = solver.evaluate({"attn_norm": RECOMPUTE})
        self.assertEqual(plan.predicted_peak_mb, 100 + 5 * 4 * 2 + 20 * 2)

    def test_no_over_budget_escape_hatch(self):
        with self.assertRaisesRegex(NoFeasiblePlanError, "no unsafe plan"):
            self.solver(budget=99).solve()

    def test_physical_limit_wins_over_user_budget(self):
        solver = self.solver(budget=200)
        solver.memory = replace(solver.memory, reserve_mb=60, extra_budget_mb=100000)
        self.assertLessEqual(solver.solve().predicted_peak_mb, 140)

    def test_full_search_contains_405_valid_combinations(self):
        groups = {name: GroupProfile(10, 2, 4, 1, 1) for name in BACKWARD_ORDER}
        modules = {name: ModuleProfile(3, 2, 10) for name in MODULE_GROUPS}
        solver = self.solver(groups=groups, modules=modules)
        self.assertEqual(len(list(solver.candidate_decisions())), 405)
        self.assertEqual(solver.solve().candidates, 405)

    def test_recompute_is_atomic_but_keep_and_offload_can_mix(self):
        groups = {name: GroupProfile(10, 2, 4, 1, 1) for name in MODULE_GROUPS["attention"]}
        solver = self.solver(groups=groups, modules={"attention": ModuleProfile(3, 2, 10)})
        decisions = {"qkv_linear": KEEP, "core_attn": OFFLOAD, "attn_proj": KEEP}
        solver.evaluate(decisions)
        decisions["core_attn"] = RECOMPUTE
        with self.assertRaisesRegex(ValueError, "entire measured"):
            solver.evaluate(decisions)

    def test_coupled_recompute_cost_is_charged_once(self):
        groups = {name: GroupProfile(10, 2, 4, 1, 1) for name in MODULE_GROUPS["attention"]}
        solver = self.solver(groups=groups, modules={"attention": ModuleProfile(7, 2, 10)}, layers=2)
        plan = solver.evaluate(dict.fromkeys(groups, RECOMPUTE))
        self.assertEqual(plan.recompute_ms, 14)

    def test_mixed_keep_protects_release_buffer_budget(self):
        groups = {"qkv_linear": GroupProfile(10, 2, 4, 1, 1),
                  "core_attn": GroupProfile(10, 2, 4, 1, 1, released_mb=30)}
        solver = self.solver(groups=groups, modules={})
        self.assertEqual(solver.evaluate({"qkv_linear": KEEP, "core_attn": OFFLOAD}).predicted_peak_mb, 140)

    def test_reload_queue_contention_is_modeled(self):
        groups = {"attn_proj": GroupProfile(10, 1, 1, 0, 10),
                  "core_attn": GroupProfile(10, 1, 1, 0, 10),
                  "qkv_linear": GroupProfile(10, 1, 1, 0, 10)}
        solver = self.solver(groups=groups, modules={}, adjacent_prefetch=2, cross_layer_prefetch=False)
        plan = solver.evaluate(dict.fromkeys(groups, OFFLOAD))
        self.assertEqual(plan.reload_stall_ms, 28)

    def test_prefetch_uses_compute_of_other_groups(self):
        groups = {"attn_proj": GroupProfile(10, 1, 20, 0, 1),
                  "core_attn": GroupProfile(10, 1, 1, 0, 10)}
        enabled = self.solver(groups=groups, modules={}, adjacent_prefetch=1, cross_layer_prefetch=False)
        disabled = self.solver(groups=groups, modules={}, adjacent_prefetch=0, cross_layer_prefetch=False)
        decisions = dict.fromkeys(groups, OFFLOAD)
        self.assertLess(enabled.evaluate(decisions).reload_stall_ms, disabled.evaluate(decisions).reload_stall_ms)

    def test_recompute_compute_can_overlap_prefetched_transfer(self):
        groups = {"mlp_norm": GroupProfile(10, 1, 1, 0, 1),
                  "moe_act": GroupProfile(10, 1, 1, 0, 1),
                  "attn_norm": GroupProfile(10, 1, 1, 0, 15)}
        solver = self.solver(groups=groups, modules={"mlp": ModuleProfile(20, 0, 0)}, adjacent_prefetch=1)
        plan = solver.evaluate({"mlp_norm": OFFLOAD, "moe_act": RECOMPUTE, "attn_norm": OFFLOAD})
        self.assertEqual(plan.reload_stall_ms, 1)

    def test_last_layer_strategy_changes_reload_schedule(self):
        solver = self.solver(modules={}, layers=2, last_layer_keep=True, adjacent_prefetch=0, cross_layer_prefetch=False)
        plan = solver.evaluate({"attn_norm": OFFLOAD})
        self.assertEqual(plan.reload_stall_ms, 4)

    def test_missing_groups_in_dense_layers(self):
        groups = {"attn_norm": GroupProfile(10, 1, 2, 1, 1),
                  "moe_act": GroupProfile(20, 2, 4, 1, 1)}
        solver = self.solver(groups=groups, modules={})
        solver.schedule = replace(solver.schedule, layers=(("attn_norm",), ("attn_norm", "moe_act")))
        self.assertEqual(solver.evaluate(dict.fromkeys(groups, KEEP)).predicted_peak_mb, 140)

    def test_selection_matches_independent_brute_force(self):
        generator = random.Random(1234)
        for _ in range(12):
            groups = {name: GroupProfile(generator.randrange(10, 80), 2, 4,
                                        generator.uniform(0.1, 8), generator.uniform(0.1, 8))
                      for name in ("attn_norm", "mlp_norm")}
            modules = {name: ModuleProfile(generator.uniform(0.2, 10), 2, 5) for name in groups}
            solver = self.solver(groups=groups, modules=modules, budget=180, inflight=2)
            feasible = []
            for actions in itertools.product((KEEP, OFFLOAD, RECOMPUTE), repeat=2):
                plan = solver.evaluate(dict(zip(groups, actions)))
                if plan.predicted_peak_mb <= plan.memory_limit_mb:
                    feasible.append(plan)
            selected = solver.solve()
            self.assertAlmostEqual(selected.predicted_overhead_ms, min(plan.predicted_overhead_ms for plan in feasible))

    def test_invalid_measurements_and_safety_factors(self):
        for value in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.solver(groups={"attn_norm": GroupProfile(value, 1, 2, 1, 1)})
        with self.assertRaises(ValueError):
            UnifiedOffloadOptimizer({}, {}, MemoryBudget(0, 100, 0, activation_margin=0.5), ScheduleConfig(((),), 1))

    def test_deterministic_ties_and_input_order(self):
        groups = {"attn_norm": GroupProfile(10, 1, 2, 0, 0), "mlp_norm": GroupProfile(10, 1, 2, 0, 0)}
        first = self.solver(groups=groups, modules={}).solve()
        second = self.solver(groups=dict(reversed(list(groups.items()))), modules={}).solve()
        self.assertEqual(first.decisions, second.decisions)


if __name__ == "__main__":
    unittest.main()
