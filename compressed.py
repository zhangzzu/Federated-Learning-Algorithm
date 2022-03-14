import torch
import copy
import numpy as np
# from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt


# 设置单个客户端的稀疏率
def prune(w, device):
    w_compressed = {name: torch.zeros(value.shape).to(
        device) for name, value in w.items()}
    for name in w.keys():
        # kmeans_c(w[name].data.clone())
        if(name.find('bias') < 0):
            w_compressed[name].data = compress_fun(
                w[name].data.clone(), device)
        else:
            w_compressed[name].data = w[name].data.clone()

    return w_compressed


# bias 不压缩
def compress_fun(T, device):
    # T>=v 的值 替换为mean，其他替换为0
    T_abs = torch.abs(T)
    T_num = T_abs.numel()
    T_top, T_pos = torch.topk(
        T_abs.flatten(), T_num-int(T_num*0.9), largest=False)
    org_shape = T.shape
    T = T.flatten()
    for i in T_pos.tolist():
        T[i] = torch.Tensor([0.0]).to(device)
    # out_ = torch.where(T >= 1, T, torch.Tensor([0.0]).to(device))
    # # 同上，其他替换位out_中对应位置的值
    # out = torch.where(T <= -1, T, out_)
    out = T.reshape(org_shape)
    return out


# def kmeans_c(x):
#     tensor_len = len(x.size())
#     if(tensor_len == 1):
#         return
#     x = x.view(x.size()[0], -1).view(-1, 1)

#     features=np.column_stack((x,np.ones(len(x))))
#     whitened = whiten(features)
#     codebook, distortion = kmeans(whitened, 100)
#     print(distortion)

#     plt.scatter(features[:, 0], features[:, 1])
#     # plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
#     plt.show()


def kmeans1(x, ncluster, niter=10):
    # while(len(x.size()) > 1):
    #     x = x.view(x.size()[0], -1)
    if(len(x.size()) > 1):
        return
    N = x.size()[0]
    c = x[torch.randperm(N)[:ncluster]]  # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        a = ((x[:, None] - c[None, :])**2).sum(-1).argmin(0)
        # move each codebook element to be the mean of the pixels that assigned to it
        c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        print('done step %d/%d, re-initialized %d dead clusters' %
              (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead clusters
    return c
