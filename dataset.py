import numpy as np
from torchvision import datasets, transforms
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, dataset

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


def iid_data(client_num):

    train_dataset1 = train_dataset['mnist']
    # 训练数据的标签
    targets = train_dataset1.targets
    # 训练数据的分类
    classes = train_dataset1.classes
    # 字典类型 key:训练数据分类 value:该分类在训练数据中的位置
    sort_datasets = {}

    for i in range(len(classes)):
        sort_datasets[i] = []

    for i in range(len(targets)):
        sort_datasets[int(targets[i])].append(i)
        # sort[targets[i]].append(i)

    # client 对应的数据字典 key:client的编号 value:该client中iid数据的位置
    dict_users = {}
    for i in range(len(classes)):
        dict_users[i] = []
        for j in range(len(sort_datasets)):
            # 分类均分，将每一类数据均分到client的数据集中
            avg_num = len(sort_datasets[j])//client_num
            #获取对应分类的value，并将对应的avg_num区间数据切片、拼接到dict_users中
            l = sort_datasets[j]
            l = l[i*avg_num:(i+1)*avg_num-1]
            dict_users[i].extend(l)

    return dict_users


def mnist_iid(num_users, traning_dataset):
    """
    Sample I.I.D. client data from MNIST dataset
    """
    data_len = len(train_dataset[traning_dataset])
    num_items = int(data_len/num_users)
    dict_users, all_idxs = {}, [i for i in range(data_len)]
    # 将数据按num_users的数量平均划分，放到dict_users数组中
    # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等
    # np.random.choice（all_idxs：数据源；num_items：要取出的数据个数；replace=False表示不可以取相同数字）
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(
            all_idxs, num_items, replace=False))
        # 把取到的数据从原数据集中去除
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    # 生成一个[0...num_shards]长度的数组
    idx_shard = [i for i in range(num_shards)]
    # 生成空数组 数量num_users
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # 生成一个60000的数组
    idxs = np.arange(num_shards*num_imgs)
    # 得到数据集的label标签 numpy 格式
    labels = train_dataset.train_labels.numpy()

    # sort labels
    # 将两个数组[1,2][3,4]  合并成[[1,2][3,4]]
    idxs_labels = np.vstack((idxs, labels))
    # [1,4,3,-1,6,9].argsort() 返回[3,0,2,1,4,5]
    # 第一个数组从小到大 对应的index
    # idxs_labels[1, :].argsort()作为前面数组的索引，idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs_labels 按后面的idex取值
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        # 从idx_shard 数组里面随机选取两个数字
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # 把随机选取的2个数字从数组中去除。
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # 通过随机数，每次取出300个，分两次拼接
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def noniid_data(indx):
    dataset = DatasetSplit(train_dataset['mnist'], indx)
    data = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return data


def mnist_noniid_a(num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = train_dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    dataset = DatasetSplit(train_dataset, dict_users)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size)


def data_test(traing_dataset):
    if(traing_dataset == "mnist"):
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
    elif(traing_dataset == "cifar10"):
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, shuffle=True, batch_size=batch_size)


if __name__ == "__main__":
    data = mnist_noniid(3)
    print(data)
