# encoding utf-8
from torch.utils.data import DataLoader
from .transforms import build_transforms
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler
from .collate_batch import train_collate_fn, val_collate_fn

def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.num_workers
    dataset = init_dataset(cfg.DATASET_NAME, root=cfg.DATASET_ROOT_DIR)
    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    train_loader = DataLoader(
        train_set, batch_size=cfg.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(dataset.train, cfg.IMS_PER_BATCH, cfg.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=train_collate_fn
    )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST_IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes