import os
import glob
import logging
import numpy as np

from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

import config

def get_transforms(mode="train"):
    shared_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=config.PIXDIM,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config.A_MIN,
            a_max=config.A_MAX,
            b_min=config.B_MIN,
            b_max=config.B_MAX,
            clip=True,
        ),
    ]

    if mode == "train":
        return Compose(
            shared_transforms + [
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=config.PATCH_SIZE,
                    pos=1, neg=1, num_samples=4,
                    image_key="image", image_threshold=0,
                ),
                RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
    elif mode == "val":
        return Compose(shared_transforms + [EnsureTyped(keys=["image", "label"])])
    else:
        raise ValueError(f"Invalid transform mode: {mode}")


def get_data_loaders():
    logging.info("--- Preparing Data ---")

    train_images = sorted(glob.glob(os.path.join(config.DATA_DIR, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(config.DATA_DIR, "labelsTr", "*.nii.gz")))

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    logging.info(f"Found {len(data_dicts)} subjects.")

    np.random.seed(config.SEED)
    np.random.shuffle(data_dicts)

    num_total = len(data_dicts)
    num_train = int(num_total * config.TRAIN_RATIO)
    num_val = int(num_total * config.VAL_RATIO)

    train_files = data_dicts[:num_train]
    val_files = data_dicts[num_train : num_train + num_val]
    test_files = data_dicts[num_train + num_val :]

    logging.info(f"Training samples: {len(train_files)}")
    logging.info(f"Validation samples: {len(val_files)}")
    logging.info(f"Testing samples: {len(test_files)}")

    logging.info("--- Creating Datasets and DataLoaders ---")

    train_transforms = get_transforms(mode="train")
    val_transforms = get_transforms(mode="val")

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=config.NUM_WORKERS
    )
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=list_data_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader, test_files