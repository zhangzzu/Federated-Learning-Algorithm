from utils import get_up_bits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch import nn
import torch
import copy
import torch.optim as optim
from dataset import mnist_iid, training_data, mnist_test
import torch.nn.functional as F

from client import Client
from algorithm import FedAvg
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Server(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """

    def __init__(self, args):
        super(Server, self).__init__()
        self.epoch = args.epochs
        self.device = args.device
        self.num_clients = args.num_clients
        self.dataset = args.dataset
        self.model = args.model

    def train(self):
        dict_users = mnist_iid(self.num_clients)
        w_locals = []
        loss_locals = []

        loss_train = []
        train_clients = [Client(copy.deepcopy(self.model), training_data(dict_users[i]), 1, self.device)
                         for i in range(self.num_clients)]
        model_up_size = 0
        for iter in range(self.epoch):
            w_locals = []
            for idx in range(self.num_clients):
                # train_client = Client(copy.deepcopy(self.model),
                #                       noniid_data(dict_users[idx]), 1, self.device)
                w, loss = train_clients[idx].train()
                print("client ", idx, "loss is ", loss)
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                model_up_size += get_up_bits(w)

            w_glob = FedAvg(w_locals)

            # copy weight to model
            self.model.load_state_dict(w_glob)

            for i in range(self.num_clients):
                train_clients[i].recv(w_glob.copy())
            # torch.save(self.model.state_dict(), 'model.pt')
            # print("w_glob size:", os.path.getsize('model.pt'))

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

        # train_clients.clear()
        print("sent to sever size:", (model_up_size/8)/1024, "kb")

        # plt.figure()
        # plt.plot(range(len(loss_train)), loss_train)
        # plt.ylabel('train_loss')
        # plt.show()
        # w_glob = Server.FedAvg(w_locals)
        # self.model.load_state_dict(w_glob)

    def test_img(self):
        # 仅仅是推论出结果，不改变该层次中的各个参数
        self.model.eval()
        # testing
        test_loss = 0
        correct = 0
        data_loader = mnist_test()
        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            if self.device.type == 'cuda':
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs,
                                         target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)
                                 ).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
        # torch.cuda.empty_cache()
