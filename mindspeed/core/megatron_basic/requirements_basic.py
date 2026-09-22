from functools import wraps
import math
import torch


def version_wrapper(fn):
    @wraps(fn)
    def wrapper(name, *args, **kwargs):
        return '2.2.0' if name == 'transformer-engine' else fn(name, *args, **kwargs)

    return wrapper


def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    return op(noop_flag_buffer, tensor_lists, *args)


def multi_tensor_l2norm(overflow_buf, tensor_lists, per_parameter):
    total_norm = 0.0
    norm_type = 2.0
    ret_per_tensor = [] if per_parameter else None
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm ** norm_type
        if per_parameter:
            ret_per_tensor.append(total_norm.clone())
    if not tensor_lists:
        grad_norm = torch.cuda.FloatTensor([0])
        total_norm = grad_norm ** norm_type
    return total_norm ** (1 / norm_type), ret_per_tensor


def multi_tensor_scale(overflow_buf, tensor_lists, scale):
    if len(tensor_lists) != 2:
        raise AssertionError('The size of tensor list must be 2, but got {}'.format(len(tensor_lists)))
    if len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError('The size of tensor list must be same, but got {} and {}'
                             .format(len(tensor_lists[0]), len(tensor_lists[1])))
    with torch.no_grad():
        for i in range(len(tensor_lists[0])):
            tensor_lists[1][i].copy_(tensor_lists[0][i] * scale)


def type_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if isinstance(res, str):
            res = res.replace('npu', 'cuda')
        return res

    return wrapper


def ensure_contiguous_wrapper(fn):
    @wraps(fn)
    def wrapper(tensor, *args, **kwargs):
        tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
        return fn(tensor, *args, **kwargs)

    return wrapper


def lcm(a, b):
    return (a * b) // math.gcd(a, b)


def dummy_function(*args, **kwargs):
    pass


def torch_all_reduce_double_dtype_bypass_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if torch.is_tensor(args[0]) and args[0].dtype == torch.double:
            args = list(args)
            args[0] = args[0].float()
            handle = fn(*args, **kwargs)
            if handle is not None:
                handle.wait()
            args[0] = args[0].double()
            return None

        return fn(*args, **kwargs)

    return wrapper


def dummy_compile(*args, **kwargs):
    if len(args) > 0 and callable(args[0]):
        def wrapper(*fn_args, **fn_kwargs):
            return args[0](*fn_args, **fn_kwargs)
        return wrapper
    else:
        def compile_wrapper(fn):
            def wrapper(*fn_args, **fn_kwargs):
                return fn(*fn_args, **fn_kwargs)
            return wrapper
        return compile_wrapper


def linear_with_grad_accum_forward_wrapper(fn):
    """Wrapper for LinearWithGradAccumulationAndAsyncCommunication.forward.

    The fine_grained_activation_offload feature installs a global hook via
    torch._C._autograd._push_saved_tensors_default_hooks that intercepts every
    ctx.save_for_backward call.  When weight (an nn.Parameter) is passed to
    save_for_backward, the hook offloads it to CPU and returns a plain Tensor on
    retrieval — losing custom attributes like 'main_grad' that
    gradient_accumulation_fusion requires.

    PyTorch's autograd engine always casts the on_get_saved_tensor return value
    to a plain Tensor, so there is no way to recover the Parameter type inside
    the hook.  The only correct fix is to keep weight out of save_for_backward
    entirely: store it directly on ctx so it bypasses the hook system.

    Strategy: shadow ctx.save_for_backward with a Python-level instance attribute
    that filters out the weight tensor before delegating to the real C++ method.
    Python attribute lookup checks instance __dict__ before the type's C-level
    descriptors, so this shadowing works reliably.
    """
    @wraps(fn)
    def wrapper(ctx, input, weight, *args, **kwargs):
        _real_save = ctx.save_for_backward

        def _save_without_weight(*tensors):
            # Filter out the weight Parameter; save only the remaining tensors
            # (typically just `input`) through the normal hook-intercepted path.
            filtered = tuple(t for t in tensors if t is not weight)
            _real_save(*filtered)
            # Store weight directly on ctx, bypassing the offload hook system.
            ctx.weight = weight

        # Shadow the C++ method with our Python function at the instance level.
        ctx.save_for_backward = _save_without_weight
        try:
            result = fn(ctx, input, weight, *args, **kwargs)
        finally:
            # Remove the shadow so ctx is clean for any subsequent use.
            try:
                del ctx.save_for_backward
            except AttributeError:
                pass
        return result
    return wrapper


class _LinearGradAccumulationContext:
    __slots__ = ('_real', '_combined')

    def __init__(self, real_ctx, saved_tensors):
        object.__setattr__(self, '_real', real_ctx)
        object.__setattr__(self, '_combined', saved_tensors)

    @property
    def saved_tensors(self):
        return object.__getattribute__(self, '_combined')

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_real'), name)

    def __setattr__(self, name, value):
        if name in ('_real', '_combined'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_real'), name, value)


def linear_with_grad_accum_backward_wrapper(fn):
    """Wrapper for LinearWithGradAccumulationAndAsyncCommunication.backward.

    Retrieves weight from ctx.weight (set by the patched forward) instead of
    ctx.saved_tensors, so that all Parameter attributes (main_grad, allreduce,
    grad_added_to_main_grad, zero_out_wgrad, etc.) are intact.

    Strategy: the original backward does ``input, weight = ctx.saved_tensors``.
    Since we only saved `input` via save_for_backward, ctx.saved_tensors is a
    1-tuple.  We unpack it ourselves and pass the real weight from ctx.weight,
    then call the original backward body directly — avoiding any need to mutate
    the read-only saved_tensors property.
    """
    @wraps(fn)
    def wrapper(ctx, grad_output):
        if not hasattr(ctx, 'weight'):
            # Patch not active (e.g. fine_grained_activation_offload disabled);
            # fall through to the original backward unchanged.
            return fn(ctx, grad_output)

        # Reconstruct the (input, weight) pair the original backward expects.
        weight = ctx.weight
        (input,) = ctx.saved_tensors  # only input was saved via save_for_backward

        return fn(_LinearGradAccumulationContext(ctx, (input, weight)), grad_output)
    return wrapper
