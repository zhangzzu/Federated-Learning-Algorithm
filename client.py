
from torch import nn
import torch.optim as optim
import copy


class Client(nn.Module):
    """ Client of Federated Learning framework.
    1. Receive global and dataset model from server
    2. Perform local training (compute gradients)
    3. Return local model (gradients) to server
    """

    def __init__(self, model, data, local_epoch, device):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(Client, self).__init__()
        self.device = device
        self.data = data
        self.model = model
        self.local_epoch = local_epoch

    def train(self):
        '''
        经过训练forward and backward 得到的是定义model的矩阵参数
        test 输入和矩阵参数运算
        '''
        # 交叉熵loss y*log p(y)
        criterion = nn.CrossEntropyLoss()
        # sgd 随机梯度下降方法 超参数0.01,momentum 冲量 跳出局部优化
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.5)
        self.model.train()
        epoch_loss = []
        for _ in range(self.local_epoch):
            batch_loss = []
            for _, dataset in enumerate(self.data, 0):
                inputs, labels = dataset
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                # 方法会更新所有的参数
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.model.state_dict(), sum(epoch_loss)/len(epoch_loss)

    def recv(self, model_param):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_param))
