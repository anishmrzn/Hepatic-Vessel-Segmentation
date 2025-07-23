import os
import logging
import time
import torch

from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import DynUNet
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

import config
from data_loader import get_data_loaders
from plot_utils import save_plots

def main():
    logging.basicConfig(level=logging.INFO)
    set_determinism(seed=config.SEED)
    os.makedirs(config.RUN_DIR, exist_ok=True)

    logging.info(f"Using device: {config.DEVICE}")

    train_loader, val_loader, _ = get_data_loaders()

    logging.info("--- Initializing Model, Loss, and Optimizer ---")

    model = DynUNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        kernel_size=config.KERNELS, strides=config.STRIDES,
        upsample_kernel_size=config.STRIDES[1:],
        norm_name="instance", res_block=True,
    ).to(config.DEVICE)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE)

    start_epoch = 0
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    iou_values = []

    if os.path.exists(config.CHECKPOINT_PATH):
        logging.info(f"--- Loading checkpoint from {config.CHECKPOINT_PATH} ---")
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_metric = checkpoint["best_metric"]
        best_metric_epoch = checkpoint["best_metric_epoch"]
        epoch_loss_values = checkpoint["epoch_loss_values"]
        metric_values = checkpoint["metric_values"]
        iou_values = checkpoint["iou_values"]
        logging.info(f"--- Resuming training from epoch {start_epoch} ---")
    else:
        logging.info("--- No checkpoint found, starting from scratch ---")

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")

    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    val_inferer = SlidingWindowInferer(
        roi_size=config.INFERER_ROI_SIZE,
        sw_batch_size=config.INFERER_SW_BATCH_SIZE,
        overlap=config.INFERER_OVERLAP,
    )

    logging.info("--- Starting Training ---")

    for epoch in range(start_epoch, config.MAX_EPOCHS):
        epoch_start_time = time.time()
        logging.info(f"Epoch {epoch + 1}/{config.MAX_EPOCHS}")

        model.train()
        epoch_loss = 0
        for step, batch_data in enumerate(train_loader):
            inputs, labels = (
                batch_data["image"].to(config.DEVICE),
                batch_data["label"].to(config.DEVICE),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"  {step+1}/{len(train_loader)}, Train_loss: {loss.item():.4f}", end='\r')

        epoch_loss /= (step + 1)
        epoch_loss_values.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % config.VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(config.DEVICE)
                    val_labels = val_data["label"].to(config.DEVICE)
                    val_outputs = val_inferer(val_inputs, model)

                    val_outputs_list = decollate_batch(val_outputs)
                    val_labels_list = decollate_batch(val_labels)

                    val_output_converted = [post_pred(p) for p in val_outputs_list]
                    val_label_converted = [post_label(l) for l in val_labels_list]

                    dice_metric(y_pred=val_output_converted, y=val_label_converted)
                    iou_metric(y_pred=val_output_converted, y=val_label_converted)

                mean_dice = dice_metric.aggregate().item()
                mean_iou = iou_metric.aggregate().item()
                dice_metric.reset()
                iou_metric.reset()

                metric_values.append(mean_dice)
                iou_values.append(mean_iou)

                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), config.BEST_MODEL_PATH)
                    logging.info("Saved new best metric model")

                logging.info(
                    f"Current epoch: {epoch + 1}, "
                    f"Current mean dice: {mean_dice:.4f}, "
                    f"Current mean IoU: {mean_iou:.4f}"
                )
                logging.info(
                    f"Best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric,
                    'best_metric_epoch': best_metric_epoch,
                    'epoch_loss_values': epoch_loss_values,
                    'metric_values': metric_values,
                    'iou_values': iou_values,
                }, config.CHECKPOINT_PATH)
                logging.info(f"Saved checkpoint to {config.CHECKPOINT_PATH}")

        logging.info(f"Time for epoch {epoch + 1}: {time.time() - epoch_start_time:.2f}s")

    logging.info("--- Training Finished ---")
    logging.info(f"Final best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

    logging.info(f"Saving plots to {config.PLOT_DIR}")
    save_plots(epoch_loss_values, metric_values, iou_values, config.VAL_INTERVAL, config.PLOT_DIR)
    logging.info("--- All tasks complete ---")

if __name__ == "__main__":
    main()