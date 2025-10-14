import functools
import torch


def _is_eligible(x):
    return x.is_floating_point() and x.is_cuda and (x.dtype is not torch.float64)


def _cast(x, dtype):
    if isinstance(x, torch.Tensor) and _is_eligible(x):
        return x.to(dtype)
    elif isinstance(x, map):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in x.items()}
    elif isinstance(x, list) or isinstance(x, tuple):
        return type(x)(map(lambda y: _cast(y, dtype), x))
    return x


def custom_fwd(fwd):
    """Wrap a custom autograd function that always uses autocast dtype."""

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if torch.is_autocast_enabled():
            with torch.autocast(device_type="cuda", enabled=False):
                dtype = torch.get_autocast_gpu_dtype()
                return fwd(*_cast(args, dtype), **_cast(kwargs, dtype))
        return fwd(*args, **kwargs)
    return decorate_fwd


def custom_bwd(bwd):
    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with torch.autocast(device_type="cuda", enabled=False):
            return bwd(*args, **kwargs)
    return decorate_bwd
