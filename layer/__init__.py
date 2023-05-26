# encoding utf-8
import torch.nn.functional as F
from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth


def make_loss(cfg, num_classes):
    triplet = TripletLoss(cfg.MARGIN)  # margin=0.3
    if cfg.IF_LABELSMOOTH == 'on':  # 标签平滑
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # 交叉熵
        print("label smooth on, numclasses:", num_classes)

    # softmax_triplet
    def loss_func(score, feat, target):
        '''
        :param score: model, FC层输出的结果，可用于表征学习
        :param feat:  model池化后的，可用于度量学习
        :param target:  标签
        :return: 损失函数由两部分组成，交叉熵+三元组。表征+度量
        '''
        if cfg.IF_LABELSMOOTH == 'on':
            return xent(score, target) + triplet(feat, target)[0]
        else:
            return F.cross_entropy(score, target) + triplet(feat, target)[0]
    return loss_func