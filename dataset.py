from distutils.command.clean import clean
from wsgiref.headers import tspecials
import numpy as np
from torch import classes
from torchvision import datasets, transforms
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, dataset
import copy

# 批处理大小
batch_size = 10

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081,))])


train_dataset = {
    'mnist': datasets.MNIST(
        root='./data', train=True, download=True, transform=transform),
    'cifar10': datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)}


# test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        datas, labels = self.dataset[self.idxs[item]]
        return datas, labels


def sort_classes():
    train_dataset1 = train_dataset['mnist']
    # 训练数据的标签
    targets = train_dataset1.targets
    # 训练数据的分类
    classes = train_dataset1.classes

    sort_datasets = {}

    for i in range(len(classes)):
        sort_datasets[i] = []

    for i in range(len(targets)):
        sort_datasets[int(targets[i])].append(i)
        # sort[targets[i]].append(i)

    return sort_datasets, classes


def data_split(data, num_split):
    # 将分类数据中的每一类分割成num_split份
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst


def choose_digit(split_data_lst, y):
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        lst = np.random.choice(available_digit, y, replace=False).tolist()
    except:
        print(available_digit)
    return lst


def iid_data(client_num):

    sort_datasets, classes = sort_classes()

    # client 对应的数据字典 key:client的编号 value:该client中iid数据的位置
    dict_users = {}
    for i in range(client_num):
        dict_users[i] = []
        for j in range(len(sort_datasets)):
            # 分类均分，将每一类数据均分到client的数据集中
            avg_num = len(sort_datasets[j])//client_num
            # 获取对应分类的value，并将对应的avg_num区间数据切片、拼接到dict_users中
            l = sort_datasets[j]
            l = l[i*avg_num:(i+1)*avg_num-1]
            dict_users[i].extend(l)

    return dict_users


def non_iid_data(client_num, y):
    sort_datasets, classes = sort_classes()

    split_mnist_traindata = []
    for digit in sort_datasets.values():
        split_mnist_traindata.append(data_split(digit, (y*client_num)//10))

    # client 对应的数据字典 key:client的编号 value:该client中iid数据的位置
    dict_users = {}
    for i in range(client_num):
        lst = []
        # 将split_mnist_traindata 中的分组数据 合并成一维数组放到对应的client中
        for j in split_mnist_traindata:
            if j != []:
                for k in range(len(j)):
                    lst.extend(j[k])
                # x = np.array(j)
                # x = x.flatten()
                # lst.extend(x.reshape(1, -1)[0])
        dict_users[i] = lst
        # 生成随机数 去除split_mnist_traindata 对应的分类数组
        for d in choose_digit(split_mnist_traindata, y):
            split_mnist_traindata[d].pop()

    return dict_users


def training_data(indx):
    dataset = DatasetSplit(train_dataset['mnist'], indx)
    data = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return data


def data_test(traing_dataset):
    if(traing_dataset == "mnist"):
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
    elif(traing_dataset == "cifar10"):
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, shuffle=True, batch_size=batch_size)


if __name__ == "__main__":
    data = non_iid_data(10)
    print(data)
