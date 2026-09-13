# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps

import torch
import torch_npu

from mindspeed.args_utils import get_full_args

PROFILE_RECORD = None


def train_wrapper(train):
    @wraps(train)
    def wrapper(*args, **kwargs):
        args_ = get_full_args()
        if args_.profile:
            args_.profile_npu = True
            args_.profile = False
        else:
            args_.profile_npu = False

        is_profile = hasattr(args_, 'profile_npu') and args_.profile_npu \
                and ((torch.distributed.get_rank() in args_.profile_ranks) or (-1 in args_.profile_ranks))
        if is_profile:
            global PROFILE_RECORD
            active = args_.profile_step_end - args_.profile_step_start
            wait = max(args_.profile_step_start - 1, 0)
            warmup = 2 if args_.profile_step_start > 0 else 0

            activities = [
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ]

            with torch_npu.profiler.profile(
                activities=activities,
                schedule=torch_npu.profiler.schedule(
                    wait=wait,
                    warmup=warmup,
                    active=active,
                    repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args_.tensorboard_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
                with_flops=True
            ) as prof:
                PROFILE_RECORD = prof
                return train(*args, **kwargs)
        return train(*args, **kwargs)

    return wrapper


def train_step_wrapper(train_step):
    @wraps(train_step)
    def wrapper(*args, **kwargs):
        args_ = get_full_args()
        ret = train_step(*args, **kwargs)
        is_profile = hasattr(args_, 'profile_npu') and args_.profile_npu and (
                (torch.distributed.get_rank() in args_.profile_ranks)
                or (-1 in args_.profile_ranks)
        )
        if is_profile:
            global PROFILE_RECORD
            PROFILE_RECORD.step()
        return ret

    return wrapper