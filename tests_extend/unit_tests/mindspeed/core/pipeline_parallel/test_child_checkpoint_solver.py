import itertools
import math
import random
import unittest
from dataclasses import replace

from mindspeed.core.pipeline_parallel.adaptive_offload.unified_offload_optimizer import (
    KEEP, OFFLOAD, RECOMPUTE, ChildRecomputeProfile, GroupProfile, MemoryBudget,
    ModuleProfile, NoFeasiblePlanError, ScheduleConfig, UnifiedOffloadOptimizer,
)


PARENT = 'layer1.dense_mlp'
CHILD = PARENT + '.activation'


class TestChildCheckpointSolver(unittest.TestCase):
    def solver(self, capacity=5400, children=None, baseline=True, **schedule_options):
        schedule = dict(layers=((PARENT,),), inflight_microbatches=4,
                        adjacent_prefetch=0, cross_layer_prefetch=False,
                        recompute_factor=1, transfer_factor=1, serialized_recompute=True,
                        baseline_recompute=baseline, unified_transport=True)
        schedule.update(schedule_options)
        return UnifiedOffloadOptimizer(
            {PARENT: GroupProfile(1200, 4, 8, 30, 30, keep_mb=1280)},
            {PARENT: ModuleProfile(100, 80, 100)},
            MemoryBudget(1000, capacity, 0, activation_margin=1), ScheduleConfig(**schedule),
            module_groups={PARENT: (PARENT,)}, nested_groups={},
            child_recomputes=({CHILD: ChildRecomputeProfile(PARENT, 8, 880, 800)}
                              if children is None else children))

    def decision(self, parent=KEEP, child=KEEP):
        return {PARENT: parent, CHILD: child}

    def test_family_contains_exactly_four_compatible_choices(self):
        solver = self.solver()
        self.assertEqual(list(solver.candidate_decisions()), [
            self.decision(), self.decision(OFFLOAD), self.decision(RECOMPUTE),
            self.decision(KEEP, RECOMPUTE)])

    def test_keep_wins_when_memory_fits(self):
        plan = self.solver(6200).solve()
        self.assertEqual(plan.decisions, self.decision())
        self.assertEqual(plan.predicted_overhead_ms, 0)

    def test_child_wins_when_keep_exceeds_capacity(self):
        solver = self.solver()
        self.assertEqual(solver.evaluate(self.decision()).predicted_peak_mb, 6120)
        plan = solver.solve()
        self.assertEqual(plan.decisions, self.decision(KEEP, RECOMPUTE))
        self.assertEqual(plan.predicted_peak_mb, 5320)
        self.assertEqual(plan.predicted_overhead_ms, 8)
        self.assertEqual(plan.transfer_count, 0)
        self.assertEqual(plan.execution['checkpoint_children'], {CHILD: PARENT})

    def test_offload_wins_when_child_workspace_does_not_fit(self):
        plan = self.solver(5000).solve()
        self.assertEqual(plan.decisions, self.decision(OFFLOAD))
        self.assertEqual(plan.predicted_overhead_ms, 60)

    def test_full_recompute_remains_available_for_tighter_budget(self):
        plan = self.solver(1500).solve()
        self.assertEqual(plan.decisions, self.decision(RECOMPUTE))
        self.assertEqual(plan.recompute_ms, 100)

    def test_no_candidate_bypasses_physical_limit(self):
        with self.assertRaises(NoFeasiblePlanError):
            self.solver(900).solve()

    def test_forward_savings_are_not_whole_step_peak_savings(self):
        solver = self.solver()
        keep = solver.evaluate(self.decision()).predicted_peak_mb
        child = solver.evaluate(self.decision(KEEP, RECOMPUTE)).predicted_peak_mb
        self.assertEqual((1280 - 880) * 4, 1600)
        self.assertEqual(keep - child, 800)

    def test_parent_union_replaces_parent_bytes_in_both_baselines(self):
        for baseline in (False, True):
            with self.subTest(baseline=baseline):
                solver = self.solver(baseline=baseline)
                plan = solver.evaluate(self.decision(KEEP, RECOMPUTE))
                self.assertEqual(plan.predicted_peak_mb, 1000 + 4 * 880 + 800)

    def test_child_union_contains_resident_and_shared_inputs_once(self):
        solver = self.solver()
        solver.groups[PARENT] = replace(solver.groups[PARENT], resident_mb=80)
        solver.modules[PARENT] = replace(solver.modules[PARENT], input_mb=80)
        self.assertEqual(solver.evaluate(self.decision()).predicted_peak_mb, 6440)
        self.assertEqual(solver.evaluate(self.decision(KEEP, RECOMPUTE)).predicted_peak_mb, 5320)

    def test_reconstruction_workspace_is_not_multiplied_by_microbatches(self):
        for inflight in (1, 2, 4, 8):
            solver = self.solver(inflight_microbatches=inflight)
            plan = solver.evaluate(self.decision(KEEP, RECOMPUTE))
            self.assertEqual(plan.predicted_peak_mb, 1000 + inflight * 880 + 800)

    def test_independent_parents_sum_retention_but_serialize_workspace(self):
        first = PARENT
        second = 'layer2.dense_mlp'
        groups = {parent: GroupProfile(1200, 4, 8, 30, 30, keep_mb=1280)
                  for parent in (first, second)}
        children = {parent + '.activation': ChildRecomputeProfile(parent, 8, 880, 800)
                    for parent in groups}
        decisions = dict.fromkeys(groups, KEEP)
        decisions.update(dict.fromkeys(children, RECOMPUTE))
        for baseline, serialized in itertools.product((False, True), repeat=2):
            solver = UnifiedOffloadOptimizer(
                groups, {}, MemoryBudget(1000, 20000, 0, activation_margin=1),
                ScheduleConfig(layers=((first,), (second,)), inflight_microbatches=4,
                               baseline_recompute=baseline, serialized_recompute=serialized),
                module_groups={parent: (parent,) for parent in groups}, child_recomputes=children)
            expected = 1000 + 2 * 4 * 880 + (800 if serialized else 1600)
            self.assertEqual(solver.evaluate(decisions).predicted_peak_mb, expected)

    def test_safety_factor_and_allocator_reserve_are_preserved(self):
        solver = self.solver(6500)
        solver.memory = replace(solver.memory, activation_margin=1.1, reserve_mb=325,
                                allocator_unavailable_mb=200)
        plan = solver.evaluate(self.decision(KEEP, RECOMPUTE))
        self.assertAlmostEqual(plan.predicted_peak_mb, 1000 + 4320 * 1.1)
        self.assertEqual(plan.memory_limit_mb, 5975)

    def test_separate_activation_baseline_and_optimizer_peak(self):
        solver = self.solver()
        solver.memory = replace(solver.memory, baseline_peak_mb=6000,
                                activation_baseline_peak_mb=800)
        self.assertEqual(solver.evaluate(self.decision(KEEP, RECOMPUTE)).predicted_peak_mb, 6000)

    def test_parent_offload_and_full_recompute_reject_child_discard(self):
        for action in (OFFLOAD, RECOMPUTE):
            with self.assertRaisesRegex(ValueError, 'cannot overlap'):
                self.solver().evaluate(self.decision(action, RECOMPUTE))

    def test_child_offload_is_not_an_unvalidated_new_copy_boundary(self):
        with self.assertRaisesRegex(ValueError, 'Child recompute requires'):
            self.solver().evaluate(self.decision(KEEP, OFFLOAD))

    def test_all_registered_child_decisions_are_required(self):
        for decisions in ({PARENT: KEEP}, {PARENT: KEEP, CHILD: KEEP, 'unknown': KEEP}):
            with self.assertRaisesRegex(ValueError, 'exactly one action'):
                self.solver().evaluate(decisions)

    def test_unmeasured_children_leave_legacy_plan_identical(self):
        reference = self.solver(children={}).solve().to_dict()
        for child in (ChildRecomputeProfile(PARENT, 8, 880, 800, False),
                      ChildRecomputeProfile(PARENT, 0, 880, 800)):
            solver = self.solver(children={CHILD: child})
            self.assertEqual(solver.solve().to_dict(), reference)
            self.assertTrue(all(set(decision) == {PARENT} for decision in solver.candidate_decisions()))

    def test_empty_child_metadata_matches_original_candidate(self):
        solver = self.solver(children={})
        self.assertEqual(len(list(solver.candidate_decisions())), 3)
        self.assertEqual(solver.solve().execution, {})

    def test_multiple_child_alternatives_require_exclusivity(self):
        extra = PARENT + '.other_boundary'
        children = {CHILD: ChildRecomputeProfile(PARENT, 8, 880, 800),
                    extra: ChildRecomputeProfile(PARENT, 12, 600, 900)}
        solver = self.solver(children=children)
        choices = list(solver.candidate_decisions())
        self.assertEqual(len(choices), 5)
        self.assertTrue(all(sum(choice[name] == RECOMPUTE for name in children) <= 1
                            for choice in choices))
        with self.assertRaisesRegex(ValueError, 'joint storage measurement'):
            solver.evaluate({PARENT: KEEP, CHILD: RECOMPUTE, extra: RECOMPUTE})

    def test_bad_child_measurements_are_rejected(self):
        reference = ChildRecomputeProfile(PARENT, 8, 880, 800)
        for field in ('recompute_ms', 'retained_mb', 'workspace_mb'):
            for value in (-1, math.inf, math.nan):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.solver(children={CHILD: replace(reference, **{field: value})})
        with self.assertRaises(ValueError):
            self.solver(children={CHILD: replace(reference, measured=1)})

    def test_bad_child_names_or_parents_are_rejected(self):
        for name in ('', PARENT, None):
            with self.assertRaises(ValueError):
                self.solver(children={name: ChildRecomputeProfile(PARENT, 8, 880, 800)})
        with self.assertRaises(ValueError):
            self.solver(children={CHILD: ChildRecomputeProfile('unknown', 8, 880, 800)})

    def test_repeated_parent_is_not_mistaken_for_one_instance(self):
        for layers in (((PARENT,), (PARENT,)), ((),)):
            with self.assertRaisesRegex(ValueError, 'exactly once'):
                self.solver(layers=layers)

    def test_multigroup_checkpoint_boundary_is_rejected(self):
        base = self.solver()
        groups = dict(base.groups, sibling=GroupProfile(1, 1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, 'single-group'):
            UnifiedOffloadOptimizer(groups, base.modules, base.memory, base.schedule,
                                    module_groups={'mlp': (PARENT, 'sibling')},
                                    child_recomputes=base.child_recomputes)

    def test_child_cost_is_charged_once_with_recompute_safety_factor(self):
        solver = self.solver(recompute_factor=1.25)
        self.assertEqual(solver.evaluate(self.decision(KEEP, RECOMPUTE)).recompute_ms, 10)
        self.assertEqual(solver.evaluate(self.decision(RECOMPUTE)).recompute_ms, 125)

    def test_bounded_search_can_choose_child_recompute(self):
        solver = self.solver()
        solver.max_exact_candidates = 1
        plan = solver.solve()
        self.assertEqual(plan.decisions, self.decision(KEEP, RECOMPUTE))
        self.assertFalse(plan.search_complete)
        self.assertLessEqual(plan.predicted_peak_mb, plan.memory_limit_mb)

    def test_bounded_search_preserves_full_recompute_seed(self):
        solver = self.solver(1500)
        solver.max_exact_candidates = 1
        self.assertEqual(solver.solve().decisions, self.decision(RECOMPUTE))

    def test_exact_solver_matches_independent_family_oracle(self):
        generator = random.Random(58409)
        observed = set()
        failures = 0
        family = ((KEEP, KEEP), (OFFLOAD, KEEP), (RECOMPUTE, KEEP), (KEEP, RECOMPUTE))
        for trial in range(160):
            parents = tuple(f'layer{index}.dense_mlp' for index in range(generator.randint(1, 4)))
            groups = {parent: GroupProfile(generator.randint(20, 180), 4, 8,
                                           generator.randint(2, 25), generator.randint(2, 25),
                                           resident_mb=generator.randint(0, 20),
                                           keep_mb=generator.randint(50, 280),
                                           d2h_storage_ratio=generator.choice((1, 1.5, 2)))
                      for parent in parents}
            modules = {parent: ModuleProfile(generator.randint(12, 70), 20, 100) for parent in parents}
            children = {parent + '.activation': ChildRecomputeProfile(
                parent, generator.randint(1, 20), generator.randint(20, 100), generator.randint(20, 160))
                        for parent in parents}
            inflight = generator.randint(1, 5)
            serialized = generator.choice((True, False))
            capacity = generator.randint(900, 2400)
            memory = MemoryBudget(1000, capacity, 50, activation_margin=1.1,
                                  allocator_unavailable_mb=20, activation_baseline_peak_mb=800)
            schedule = ScheduleConfig(layers=tuple((parent,) for parent in parents),
                                      inflight_microbatches=inflight, adjacent_prefetch=0,
                                      cross_layer_prefetch=False, baseline_recompute=True,
                                      serialized_recompute=serialized, unified_transport=True,
                                      synchronous_d2h=True, d2h_slots=1,
                                      recompute_factor=1.25, transfer_factor=1.1)
            solver = UnifiedOffloadOptimizer(groups, modules, memory, schedule,
                                            module_groups={parent: (parent,) for parent in parents},
                                            nested_groups={}, child_recomputes=children)
            best = None
            for selection in itertools.product(family, repeat=len(parents)):
                decisions = {}
                retained, cost, transfer_work, transfer_count = 0.0, 0.0, 0.0, 0
                workspaces, copied, ratios = [], [], []
                for parent, (parent_action, child_action) in zip(parents, selection):
                    name = parent + '.activation'
                    profile = groups[parent]
                    decisions.update({parent: parent_action, name: child_action})
                    if child_action == RECOMPUTE:
                        retained += children[name].retained_mb
                        workspaces.append(children[name].workspace_mb)
                        cost += children[name].recompute_ms * 1.25
                    elif parent_action == KEEP:
                        retained += profile.keep_activation_mb + profile.resident_mb
                    elif parent_action == OFFLOAD:
                        retained += profile.resident_mb
                        copied.append(profile.activation_mb)
                        ratios.append(profile.d2h_storage_ratio)
                        transfer_work += (profile.d2h_ms + profile.h2d_ms) * 1.1
                        transfer_count += 2
                    else:
                        cost += modules[parent].recompute_ms * 1.25
                workspace = max(workspaces, default=0) if serialized else sum(workspaces)
                staging = max(copied, default=0) * max(ratios, default=1)
                prefetch = max(copied, default=0)
                peak = max(1000, 800 + 1.1 * (retained * inflight + workspace + staging + prefetch))
                if peak > capacity - 70:
                    continue
                key = (cost + transfer_work, transfer_work, transfer_count, peak,
                       tuple(decisions[name] for name in sorted(decisions)))
                if best is None or key < best[0]:
                    best = key, decisions
            if best is None:
                failures += 1
                with self.assertRaises(NoFeasiblePlanError):
                    solver.solve()
                continue
            plan = solver.solve()
            self.assertEqual(plan.decisions, best[1], trial)
            self.assertAlmostEqual(plan.predicted_peak_mb, best[0][3], places=8)
            self.assertAlmostEqual(plan.predicted_overhead_ms, best[0][0], places=8)
            for parent in parents:
                observed.add((plan.decisions[parent], plan.decisions[parent + '.activation']))
        self.assertEqual(observed, set(family))
        self.assertGreater(failures, 0)


if __name__ == '__main__':
    unittest.main()
