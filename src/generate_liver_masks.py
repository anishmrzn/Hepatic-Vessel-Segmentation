import SimpleITK as sitk
import numpy as np
import os
import glob
import time
from tqdm import tqdm
import config

def create_liver_mask(image_path, output_dir):
    file_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, file_name)

    if os.path.exists(output_path):
        print(f"  Mask already exists for {file_name}. Skipping.")
        return

    try:
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)

        initial_threshold_min = -10
        initial_threshold_max = 200
        
        thresholded_image = sitk.BinaryThreshold(
            image,
            lowerThreshold=initial_threshold_min,
            upperThreshold=initial_threshold_max,
            insideValue=1,
            outsideValue=0
        )
        
        connected_components = sitk.ConnectedComponent(thresholded_image)
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(connected_components)

        largest_label = 0
        largest_size = 0
        for i in label_stats.GetLabels():
            if label_stats.GetNumberOfPixels(i) > largest_size:
                largest_size = label_stats.GetNumberOfPixels(i)
                largest_label = i
        
        liver_candidate = sitk.BinaryThreshold(
            connected_components,
            lowerThreshold=float(largest_label),
            upperThreshold=float(largest_label),
            insideValue=1,
            outsideValue=0
        )

        closing_radius = (5, 5, 5)
        closed_mask = sitk.BinaryMorphologicalClosing(
            liver_candidate,
            kernelRadius=closing_radius,
            foregroundValue=1,
            safeBorder=True
        )

        final_mask = closed_mask
        final_mask.CopyInformation(image)

        sitk.WriteImage(final_mask, output_path)
        print(f"  Generated and saved liver mask for {file_name}")

    except Exception as e:
        print(f"ERROR: Failed to process {file_name}: {e}")

def main():
    print("--- Starting Liver ROI Mask Generation ---")
    print(f"Reading images from: {config.RAW_TRAIN_IMAGES_DIR}")
    print(f"Saving masks to: {config.LIVER_ROI_MASKS_DIR}")

    os.makedirs(config.LIVER_ROI_MASKS_DIR, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(config.RAW_TRAIN_IMAGES_DIR, '*.nii.gz')))
    
    if not image_files:
        print(f"No NIfTI files found in {config.RAW_TRAIN_IMAGES_DIR}. Please check your config.DATA_DIR and dataset structure.")
        return

    print(f"Found {len(image_files)} image files to process for ROI generation.")

    start_time = time.time()
    for image_path in tqdm(image_files, desc="Generating ROI Masks"):
        create_liver_mask(image_path, config.LIVER_ROI_MASKS_DIR)
    
    end_time = time.time()
    print(f"\n--- Finished Liver ROI Mask Generation ---")
    print(f"Total time taken: {(end_time - start_time):.2f} seconds")
    print(f"ROI masks saved to: {config.LIVER_ROI_MASKS_DIR}")

if __name__ == '__main__':
    main()