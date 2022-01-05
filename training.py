from _typeshed import Self

from torch.utils.data import dataset
from client import Client
from center import Server

#放到center中 center拿到epoch 在sever 中执行
class LocadTraining(object):
    def __init__(self, epoch, dataset, model):
        self.epoch = epoch
        self.dataset = dataset
        self.model = model

    def train(self):
        for _ in range(self.epoch):
            w_locals = []
            for idx in t:
                local = Client(self.dataset[idx])
                w = local.train()
                w_locals[idx] = w
            w_glob = Server.FedAvg(w_locals)
            self.model.load_state_dict(w_glob)

