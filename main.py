# Application of FL task
from models import Model
import torch as nn

from dataset import mnist_noniid
from center import Server
from options import args_parser


"""
1. load_data
2. generate clients (step 3)
3. generate aggregator
4. training
"""

if __name__ == "__main__":
    use_cuda = nn.cuda.is_available()
    device = nn.device("cuda" if use_cuda else "cpu")
    num_clients = 50
    # 训练参数
    args = args_parser()
    args.epochs = 10
    args.num_clients = num_clients
    args.model = Model().to(device)
    args.dataset = mnist_noniid(num_clients)
    args.device = device

    fl_entity = Server(args).to(device)
    fl_entity.train()
    fl_entity.test_img()
