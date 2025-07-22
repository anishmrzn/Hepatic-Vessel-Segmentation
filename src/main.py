import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import glob
import random
import warnings

import config
from data_loader import HepaticVesselDataset
from models.unet_model import UNet


def main():
    print("--- Loading Configuration ---")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Train/Val/Test Split: {config.TRAIN_RATIO*100}% / {config.VAL_RATIO*100}% / {config.TEST_RATIO*100}%")
    print(f"Patch Size: {config.PATCH_SIZE}")

    print("\n--- Preparing Data and Performing Split ---")

    all_image_paths = sorted(glob.glob(os.path.join(config.RAW_TRAIN_IMAGES_DIR, '*.nii.gz')))
    
    if not all_image_paths:
        raise FileNotFoundError(f"No image files found in {config.RAW_TRAIN_IMAGES_DIR}. Please check DATA_DIR in config.py.")

    data_tuples = []
    for img_path in all_image_paths:
        base_name = os.path.basename(img_path)
        lbl_path = os.path.join(config.RAW_TRAIN_LABELS_DIR, base_name)
        roi_path = os.path.join(config.LIVER_ROI_MASKS_DIR, base_name)

        if os.path.exists(lbl_path) and os.path.exists(roi_path):
            data_tuples.append((img_path, lbl_path, roi_path))
        else:
            warnings.warn(f"Missing files for {base_name}. Skipping volume.")
            print(f"  Image: {img_path} ({os.path.exists(img_path)})")
            print(f"  Label: {lbl_path} ({os.path.exists(lbl_path)})")
            print(f"  ROI:   {roi_path} ({os.path.exists(roi_path)})")
    
    if not data_tuples:
        raise ValueError("No complete image-label-ROI triplets found after checking. Cannot proceed with data splitting. Ensure all files exist or generate ROI masks.")

    print(f"Found {len(data_tuples)} complete volumes for splitting.")
    random.seed(42)
    random.shuffle(data_tuples)

    total_volumes = len(data_tuples)
    train_size = int(total_volumes * config.TRAIN_RATIO)
    val_size = int(total_volumes * config.VAL_RATIO)
    test_size = total_volumes - train_size - val_size

    train_data = data_tuples[0:train_size]
    val_data = data_tuples[train_size:train_size + val_size]
    test_data = data_tuples[train_size + val_size:]

    print(f"Train volumes: {len(train_data)}, Validation volumes: {len(val_data)}, Test volumes: {len(test_data)}")

    train_img_paths, train_lbl_paths, train_roi_paths = zip(*train_data) if train_data else ([], [], [])
    val_img_paths, val_lbl_paths, val_roi_paths = zip(*val_data) if val_data else ([], [], [])
    test_img_paths, test_lbl_paths, test_roi_paths = zip(*test_data) if test_data else ([], [], [])

    print("\n--- Defining Data Transforms (Augmentations only) ---")
    
    train_dataset = HepaticVesselDataset(
        image_paths=list(train_img_paths),
        label_paths=list(train_lbl_paths),
        roi_paths=list(train_roi_paths)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print(f"Train DataLoader created with {len(train_dataset)} slices and batch size {config.BATCH_SIZE}.")

    val_dataset = HepaticVesselDataset(
        image_paths=list(val_img_paths),
        label_paths=list(val_lbl_paths),
        roi_paths=list(val_roi_paths)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print(f"Validation DataLoader created with {len(val_dataset)} slices and batch size {config.BATCH_SIZE}.")

    test_dataset = HepaticVesselDataset(
        image_paths=list(test_img_paths),
        label_paths=list(test_lbl_paths),
        roi_paths=list(test_roi_paths)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print(f"Test DataLoader created with {len(test_dataset)} slices and batch size {config.BATCH_SIZE}.")


    print("\n--- Initializing Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=config.IN_CHANNELS, out_channels=config.OUT_CHANNELS).to(device)
    print("Model initialized (UNet).")
    print(f"Model will run on: {device}")


    print("\n--- Defining Loss Function and Optimizer ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    print(f"Loss function (BCEWithLogitsLoss) and Optimizer (Adam with LR={config.LEARNING_RATE}) defined.")


    print("\n--- Starting Training Loop ---")
    
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1} Training finished. Avg Loss: {epoch_loss:.4f}")

        if len(val_dataset) > 0:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for val_images, val_masks in val_loader:
                    val_images = val_images.to(device)
                    val_masks = val_masks.to(device)
                    val_outputs = model(val_images)
                    val_loss = criterion(val_outputs, val_masks)
                    val_running_loss += val_loss.item() * val_images.size(0)
            
            val_epoch_loss = val_running_loss / len(val_dataset)
            print(f"Epoch {epoch+1} Validation Loss: {val_epoch_loss:.4f}")

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                model_save_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
                torch.save(model.state_dict(), model_save_path)
                print(f"  Saved best model to {model_save_path} (Validation Loss: {best_val_loss:.4f})")

    print("\n--- Training Loop Finished ---")

if __name__ == '__main__':
    main()