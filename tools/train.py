# encoding utf-8
import argparse
from loguru import logger
import sys
sys.path.append('.')
from modeling import build_model
from data import make_data_loader
from solver import make_optimizer, WarmupMultiStepLR
from layer import make_loss
from engine.trainer import do_train


def train(args):
    # 数据集
    train_loader, val_loader, num_query, num_classes = make_data_loader(args)
    # model
    model = build_model(args, num_classes)
    # 优化器
    optimizer = make_optimizer(args, model)
    # loss
    loss_func = make_loss(args, num_classes)
    start_epoch = 0
    scheduler = WarmupMultiStepLR(optimizer, args.STEPS, args.GAMMA, args.WARMUP_FACTOR,
                                  args.WARMUP_ITERS, args.WARMUP_METHOD)
    if not args.kd:
        print('ready train~')
    else:
        print('ready kd train!')
    do_train(args,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_func,
             num_query,
             start_epoch,
             args.kd,
             args.feature_loss_coefficient)


if __name__ == '__main__':
    parse = argparse.ArgumentParser('Reid train')
    parse.add_argument('--last_stride', type=int, default=1, help='last stride')
    parse.add_argument('--model_path', type=str, default='./weights/ReID_resnet50_ibn_a.pth', help='pretrained model weight')
    parse.add_argument('--model_name', type=str, default='resnet50', help='model name')
    parse.add_argument('--neck', type=str, default='bnneck', help='no or bnneck')
    parse.add_argument('--neck_feat', type=str, default='after', help='before or after')
    parse.add_argument('--INPUT_SIZE', default=[256, 128])
    parse.add_argument('--INPUT_MEAN', default=[0.485, 0.456, 0.406], help='input image mean')
    parse.add_argument('--INPUT_STD', default=[0.229, 0.224, 0.225], help='image std')
    parse.add_argument('--PROB', default=0.5, help='RandomHorizontalFlip')
    parse.add_argument('--PADDING', default=10, help='transforms image pad')
    parse.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parse.add_argument('--DATASET_NAME', type=str, default='markt1501', help='dataset name')
    parse.add_argument('--DATASET_ROOT_DIR', default='./data', help='dataset root dir')
    parse.add_argument('--SAMPLER', default='softmax_triplet', help='sampler')
    parse.add_argument('--IMS_PER_BATCH', type=int, default=8, help='IMS_PER_BATCH')
    parse.add_argument('--TEST_IMS_PER_BATCH', type=int, default=4, help='TEST_IMS_PER_BATCH')
    parse.add_argument('--NUM_INSTANCE', type=int, default=4, help='number of instances per identity in a batch')
    # 超参数设置
    parse.add_argument('--OPTIMIZER_NAME', default='Adam', help='OPTIMIZER_NAME, Adam or SGD')
    parse.add_argument('--BASE_LR', default=0.00035, help='base lr')
    parse.add_argument('--WEIGHT_DECAY', default=5e-4, help='weight decay')
    parse.add_argument('--BIAS_LR_FACTOR', default=1, help='BIAS_LR_FACTOR')
    parse.add_argument('--WEIGHT_DECAY_BIAS', default=5e-4, help='weight decay bias')
    parse.add_argument('--MOMENTUM', default=0.9, help='MOMENTUM')
    parse.add_argument('--STEPS', default=[40, 70], help='steps')
    parse.add_argument('--GAMMA', default=0.1, help='GAMMA')
    parse.add_argument('--WARMUP_FACTOR', default=0.01, help='WARMUP_FACTOR')
    parse.add_argument('--WARMUP_ITERS', default=10, help='WARMUP_ITERS')
    parse.add_argument('--WARMUP_METHOD', type=str, default='linear', help='WARMUP_METHOD')
    # loss
    parse.add_argument('--MARGIN', default=0.3, help='TripletLoss MARGIN')
    parse.add_argument('--IF_LABELSMOOTH', default='on', help='label smooth')
    # train
    parse.add_argument('--OUTPUT_DIR', default='./logs', help='output dir')
    parse.add_argument('--DEVICE', type=str, default='cuda', help='gpu or cpu')
    parse.add_argument('--MAX_EPOCHS', type=int, default=120, help='epochs')
    # kd train
    parse.add_argument('--kd', action='store_true', default=False, help='kd train')
    parse.add_argument('--feature_loss_coefficient', type=float, default=0.03, help='kd train feature_loss_coefficient')
    arg = parse.parse_args()
    logger.info(arg)
    train(arg)
