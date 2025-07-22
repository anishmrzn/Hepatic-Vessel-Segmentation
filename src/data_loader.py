import os
import glob
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import warnings

import config
from preprocessing import ApplyFrangiVesselness, ApplyCEDFilter


class HepaticVesselDataset(Dataset):
    def __init__(self, image_paths, label_paths, roi_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.roi_paths = roi_paths
        self.data_tuples = []
        self.preprocess_volumes()


    def preprocess_volumes(self):
        print(f"Loading and pre-processing {len(self.image_paths)} volumes for slices...")
        temp_data_tuples = []

        frangi = ApplyFrangiVesselness(
            sigmas=config.FRANGI_SCALES,
            alpha=config.FRANGI_ALPHA,
            beta=config.FRANGI_BETA,
            black_ridges=config.FRANGI_BLACK_RIDGES
        )
        ced = ApplyCEDFilter(
            iterations=config.CED_ITERATIONS,
            K=config.CED_K_PARAM,
            lambda_param=config.CED_LAMBDA_PARAM,
            option=config.CED_OPTION
        )

        for i, (image_path, label_path, roi_path) in enumerate(zip(self.image_paths, self.label_paths, self.roi_paths)):
            try:
                image_itk = sitk.ReadImage(image_path)
                label_itk = sitk.ReadImage(label_path)
                roi_itk = sitk.ReadImage(roi_path)

                resample_image = sitk.ResampleImageFilter()
                resample_image.SetOutputSpacing(config.ISOTROPIC_RESOLUTION)
                resample_image.SetInterpolator(sitk.sitkLinear)
                resample_image.SetOutputOrigin(image_itk.GetOrigin())
                resample_image.SetOutputDirection(image_itk.GetDirection())
                resample_image.SetReferenceImage(image_itk)
                image_resampled = resample_image.Execute(image_itk)
                output_size = image_resampled.GetSize()

                resample_mask = sitk.ResampleImageFilter()
                resample_mask.SetOutputSpacing(config.ISOTROPIC_RESOLUTION)
                resample_mask.SetInterpolator(sitk.sitkNearestNeighbor)
                resample_mask.SetOutputOrigin(label_itk.GetOrigin())
                resample_mask.SetOutputDirection(label_itk.GetDirection())
                resample_mask.SetSize(output_size)
                resample_mask.SetReferenceImage(label_itk)
                
                label_resampled = resample_mask.Execute(label_itk)
                roi_resampled = resample_mask.Execute(roi_itk)

                image_array = sitk.GetArrayFromImage(image_resampled)
                label_array = sitk.GetArrayFromImage(label_resampled)
                roi_array = sitk.GetArrayFromImage(roi_resampled)

                valid_z_indices = np.where(roi_array.sum(axis=(1, 2)) > 0)[0]

                for z in valid_z_indices:
                    img_slice = image_array[z, :, :]
                    lbl_slice = label_array[z, :, :]
                    roi_slice = roi_array[z, :, :]

                    lbl_slice = lbl_slice * roi_slice

                    min_hu = -100.0
                    max_hu = 400.0
                    img_slice_norm = np.clip(img_slice, min_hu, max_hu)
                    img_slice_norm = (img_slice_norm - min_hu) / (max_hu - min_hu)
                    
                    original_img_tensor_chw = torch.from_numpy(img_slice_norm).float().unsqueeze(0)

                    frangi_output_tensor = frangi(original_img_tensor_chw)
                    ced_output_tensor = ced(original_img_tensor_chw)

                    combined_input_tensor = (original_img_tensor_chw + frangi_output_tensor + ced_output_tensor) / 3.0
                    combined_input_tensor = torch.clamp(combined_input_tensor, 0, 1)

                    lbl_tensor = torch.from_numpy(lbl_slice).float().unsqueeze(0)
                    
                    H, W = config.PATCH_SIZE
                    _, current_H, current_W = combined_input_tensor.shape

                    pad_H, pad_W = 0, 0
                    if current_H < H:
                        pad_H = H - current_H
                    if current_W < W:
                        pad_W = W - current_W
                    
                    if pad_H > 0 or pad_W > 0:
                        pad_left = pad_W // 2
                        pad_right = pad_W - pad_left
                        pad_top = pad_H // 2
                        pad_bottom = pad_H - pad_top
                        
                        padding_transform = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)
                        combined_input_tensor = padding_transform(combined_input_tensor)
                        lbl_tensor = padding_transform(lbl_tensor)
                        _, current_H, current_W = combined_input_tensor.shape

                    if current_H > H or current_W > W:
                        crop_transform = transforms.CenterCrop(config.PATCH_SIZE)
                        combined_input_tensor = crop_transform(combined_input_tensor)
                        lbl_tensor = crop_transform(lbl_tensor)
                    
                    temp_data_tuples.append((combined_input_tensor, lbl_tensor))
            
            except Exception as e:
                warnings.warn(f"Error processing volume {os.path.basename(image_path)}: {e}. Skipping this volume.")

        self.data_tuples = temp_data_tuples
        print(f"Finished pre-processing. Total 2D slices for this dataset: {len(self.data_tuples)}")
        
        if len(self.data_tuples) == 0:
            warnings.warn("No 2D slices were extracted for this dataset. Dataset will be empty.")


    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        image_slice, label_slice = self.data_tuples[idx]
        return image_slice, label_slice