

from statistics import mode
from turtle import up


def get_bits(T):
    # 在pytorch中32位存储一个参数
    k = T.numel()
    return k*32


def get_up_bits(w):
    update_size = sum([get_bits(T) for T in w.values()])
    return update_size


def get_sparsity_rate(w):
    # 计算模型的稀疏率
    model = []
    model_zero = []
    for T in w.values():
        k = T.numel()
        model.append(k)
        s = 0
        for i in T.flatten().tolist():
            if(i == 0):
                s += 1
        model_zero.append(s)
    return sum(model_zero)/sum(model), model, model_zero
