import os

import torch
import torch.distributed as dist


def init_distributed(launcher='none', local_rank=0):
    if launcher == 'none':
        return False
    if dist.is_available() and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    return True


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor: torch.Tensor, average=True):
    if not is_dist_avail_and_initialized():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if average:
        t /= get_world_size()
    return t
