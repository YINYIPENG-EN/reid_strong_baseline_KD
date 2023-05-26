import argparse
import sys
sys.path.append('.')
import os
from os import mkdir
from loguru import logger
from torch.backends import cudnn
from data import make_data_loader
from engine.inference import inference
from modeling import build_model


if __name__ == '__main__':
    parse = argparse.ArgumentParser('Reid test')
    parse.add_argument('--DEVICE', default='cuda')
    parse.add_argument('--DEVICE_ID', type=str, default='0')
    parse.add_argument('--RE_RANKING', default='no')
    parse.add_argument('--FEAT_NORM', default='yes')
    parse.add_argument('--OUTPUT_DIR', default='./logs')
    parse.add_argument('--num_workers', default=4)
    parse.add_argument('--DATASET_NAME', default='markt1501')
    parse.add_argument('--DATASET_ROOT_DIR', default='data')
    parse.add_argument('--IMS_PER_BATCH', default=8)
    parse.add_argument('--NUM_INSTANCE', default=4)
    parse.add_argument('--TEST_IMS_PER_BATCH', default=4)
    parse.add_argument('--INPUT_SIZE', default=[256, 128])
    parse.add_argument('--INPUT_MEAN', default=[0.485, 0.456, 0.406], help='input image mean')
    parse.add_argument('--INPUT_STD', default=[0.229, 0.224, 0.225], help='image std')
    parse.add_argument('--PROB', default=0.5, help='RandomHorizontalFlip')
    parse.add_argument('--PADDING', default=10, help='transforms image pad')
    parse.add_argument('--last_stride', type=int, default=1, help='last stride')
    parse.add_argument('--model_name', type=str, default='resnet34', help='model name')
    parse.add_argument('--model_path', type=str, default='./weights/resnet34-333f7ec4.pth',
                       help='pretrained model weight')
    parse.add_argument('--Test_weight', type=str, default='')
    parse.add_argument('--neck', type=str, default='bnneck', help='no or bnneck')
    parse.add_argument('--neck_feat', type=str, default='after', help='before or after')

    arg = parse.parse_args()
    print(arg)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    output_dir = arg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)
    logger.info("Using {} GPUS".format(num_gpus))

    if arg.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = arg.DEVICE_ID
    cudnn.benchmark = True
    train_loader, val_loader, num_query, num_classes = make_data_loader(arg)
    model = build_model(arg, num_classes)
    model.load_param(arg.Test_weight)
    inference(arg, model, val_loader, num_query)