

from turtle import up


def get_bits(T):
    #在pytorch中32位存储一个参数
    k = T.numel()
    return k*32


def get_up_bits(w):
    update_size = sum([get_bits(T) for T in w.values()])
    return update_size
