# Application of FL task
from ast import arg
from statistics import mode
from models import CNN, VGG11
import torch as nn

from dataset import non_iid_data
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
    traing_model = "CNN"
    traing_dataset = "mnist"
    num_clients = 30
    num_epochs = 10
    # 训练参数
    args = args_parser()
    args.epochs = num_epochs
    args.num_clients = num_clients
    # args.model = models.vgg11(pretrained=True).to(device)
    if(traing_model == "CNN"):
        args.model = CNN().to(device)
    elif(traing_model == "VGG"):
        arg.model = VGG11().to(device)
    args.dataset = "non-iid"  # iid or non-iid
    args.y =1
    args.device = device

    fl_entity = Server(args).to(device)
    fl_entity.train()
    fl_entity.test_img(traing_dataset)
