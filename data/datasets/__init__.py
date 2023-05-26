# encoding utf-8
from .market1501 import Market1501
from .dataset_loader import ImageDataset
__factory = {
    'markt1501': Market1501
}


def get_names():
    return __factory.keys()


# 数据集初始化
def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

