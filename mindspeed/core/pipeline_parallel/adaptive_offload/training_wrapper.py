# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Wrapper for megatron.training.training.train() to inject
# adaptive memory profiler lifecycle (on_iteration_start / on_iteration_end /
# apply_optimization_results) into the training loop.
#
# The original Megatron code injects these calls directly inside the while-loop
# of train(). Since MindSpeed uses a non-invasive patch mechanism, we wrap the
# entire train() function and inject a post-train_step hook via monkey-patching
# the inner train_step call.

from functools import wraps

import torch


def train_wrapper(original_train):
    """Wrap megatron.training.training.train to inject adaptive offload profiler."""

    @wraps(original_train)
    def wrapper(
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        train_data_iterator,
        valid_data_iterator,
        process_non_loss_data_func,
        config,
        checkpointing_context,
        non_loss_data_func,
    ):
        from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
            ADAPTIVE_OFFLOAD_ENABLED,
            PipelineOffloadManager,
        )
        from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import (
            AdaptiveMemoryProfiler,
        )

        if not ADAPTIVE_OFFLOAD_ENABLED:
            # Adaptive offload disabled — call original train without intervention.
            return original_train(
                forward_step_func,
                model,
                optimizer,
                opt_param_scheduler,
                train_data_iterator,
                valid_data_iterator,
                process_non_loss_data_func,
                config,
                checkpointing_context,
                non_loss_data_func,
            )

        # --- Set up adaptive profiler once ---
        _adaptive_profiler = AdaptiveMemoryProfiler.get_instance()

        if not getattr(_adaptive_profiler, "_layers_count_set", False):
            try:
                from megatron.training.utils import unwrap_model
                from megatron.core.transformer.transformer_block import (
                    get_num_layers_to_build,
                )
                _unwrapped = unwrap_model(model)
                _model_cfg = None
                for _m in _unwrapped:
                    if hasattr(_m, "config"):
                        _model_cfg = _m.config
                        break
                if _model_cfg is not None:
                    _num_layers = get_num_layers_to_build(_model_cfg)
                    _adaptive_profiler.set_num_layers_on_this_rank(_num_layers)
            except Exception:
                pass
            _adaptive_profiler._layers_count_set = True

        # --- Monkey-patch train_step to inject pre/post hooks ---
        import megatron.training.training as _training_module

        _original_train_step = _training_module.train_step

        def _patched_train_step(*args, **kwargs):
            # on_iteration_start is called BEFORE train_step.
            # We get the current iteration from megatron args.
            from megatron.training import get_args
            _args = get_args()
            _iteration = getattr(_args, 'curr_iteration', 0)

            _adaptive_profiler.on_iteration_start(_iteration)

            result = _original_train_step(*args, **kwargs)

            _adaptive_profiler.on_iteration_end(_iteration)

            # Apply optimization results after profiling completes.
            if (
                _adaptive_profiler.is_profiling_done()
                and not _adaptive_profiler.is_optimization_applied()
            ):
                skip_groups, recompute_groups = (
                    _adaptive_profiler.apply_optimization_results()
                )
                mgr = PipelineOffloadManager.get_instance()
                if skip_groups:
                    mgr.skip_offload_groups = skip_groups
                else:
                    mgr.skip_offload_groups = set()
                if recompute_groups:
                    mgr.recompute_groups = recompute_groups
                else:
                    mgr.recompute_groups = set()

                # Print results on relevant ranks.
                _should_print = getattr(
                    _adaptive_profiler, "_is_stall_profile_rank", False
                )
                if not _should_print:
                    _should_print = (
                        torch.distributed.is_initialized()
                        and torch.distributed.get_rank() == 0
                    )
                if _should_print:
                    _global_rank = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else 0
                    )
                    _pp_rank = _adaptive_profiler._pp_rank
                    _reopt_count = _adaptive_profiler._reoptimize_count
                    _label = (
                        f" (re-optimization #{_reopt_count})"
                        if _reopt_count > 0
                        else ""
                    )
                    if skip_groups or recompute_groups:
                        print(
                            f"[AdaptiveOffload] Applied optimization results{_label}:\n"
                            f"  Global rank         : {_global_rank}\n"
                            f"  PP rank             : {_pp_rank}\n"
                            f"  KEEP (skip offload) : {skip_groups if skip_groups else 'none'}\n"
                            f"  RECOMPUTE           : {recompute_groups if recompute_groups else 'none'}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[AdaptiveOffload] All groups have sufficient overlap{_label} "
                            f"(global_rank={_global_rank}, pp_rank={_pp_rank}), "
                            "no groups skipped or recomputed.",
                            flush=True,
                        )

            return result

        # Install the patched train_step for the duration of train().
        _training_module.train_step = _patched_train_step
        try:
            return original_train(
                forward_step_func,
                model,
                optimizer,
                opt_param_scheduler,
                train_data_iterator,
                valid_data_iterator,
                process_non_loss_data_func,
                config,
                checkpointing_context,
                non_loss_data_func,
            )
        finally:
            # Restore original train_step to avoid leaking the patch.
            _training_module.train_step = _original_train_step

    return wrapper
