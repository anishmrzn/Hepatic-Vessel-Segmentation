import os
import logging
import torch

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import DynUNet
from monai.transforms import AsDiscrete

import config
from data_loader import get_transforms

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("--- Starting Test Evaluation ---")

    from data_loader import get_data_loaders
    _, _, test_files = get_data_loaders()

    if not test_files:
        logging.warning("No test files found. Exiting.")
        return

    logging.info(f"Found {len(test_files)} test samples.")
    logging.info(f"Using device: {config.DEVICE}")

    test_transforms = get_transforms(mode="val")
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=config.NUM_WORKERS)

    logging.info(f"Loading model from: {config.MODEL_PATH}")

    model = DynUNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        kernel_size=config.KERNELS, strides=config.STRIDES,
        upsample_kernel_size=config.STRIDES[1:],
        norm_name="instance", res_block=True,
    ).to(config.DEVICE)

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")

    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    test_inferer = SlidingWindowInferer(
        roi_size=config.INFERER_ROI_SIZE,
        sw_batch_size=config.INFERER_SW_BATCH_SIZE,
        overlap=config.INFERER_OVERLAP,
    )

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            logging.info(f"Evaluating sample {i+1}/{len(test_loader)}")
            test_inputs = test_data["image"].to(config.DEVICE)
            test_labels = test_data["label"].to(config.DEVICE)

            test_outputs = test_inferer(test_inputs, model)

            test_outputs_list = decollate_batch(test_outputs)
            test_labels_list = decollate_batch(test_labels)

            test_output_converted = [post_pred(p) for p in test_outputs_list]
            test_label_converted = [post_label(l) for l in test_labels_list]

            dice_metric(y_pred=test_output_converted, y=test_label_converted)
            iou_metric(y_pred=test_output_converted, y=test_label_converted)

    mean_dice = dice_metric.aggregate().item()
    mean_iou = iou_metric.aggregate().item()

    logging.info("--- Test Evaluation Finished ---")
    logging.info(f"Mean Dice on Test Set: {mean_dice:.4f}")
    logging.info(f"Mean IoU on Test Set: {mean_iou:.4f}")

if __name__ == "__main__":
    main()