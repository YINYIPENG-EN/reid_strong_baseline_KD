import torchvision.transforms as T
from .transforms import RandomErasing

# 建立数据预处理规则
def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT_MEAN, std=cfg.INPUT_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT_SIZE),
            T.RandomHorizontalFlip(p=cfg.PROB),  # 0.5
            T.Pad(cfg.PADDING),  # 10
            T.RandomCrop(cfg.INPUT_SIZE),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, mean=cfg.INPUT_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT_SIZE),
            T.ToTensor(),
            normalize_transform
        ])
    return transform