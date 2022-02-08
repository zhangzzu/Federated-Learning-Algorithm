import torch
import copy


def prune(w, device):
    w_compressed = {name: torch.zeros(value.shape).to(
        device) for name, value in w.items()}
    for name in w.keys():
        w_compressed[name].data = compress_fun(w[name].data.clone(), device)

    return w_compressed


def compress_fun(T, device):
    # T>=v 的值 替换为mean，其他替换为0
    out_ = torch.where(T >= 0.5, T, torch.Tensor([0.0]).to(device))
    # 同上，其他替换位out_中对应位置的值
    out = torch.where(T <= -0.5, T, out_)

    return out
