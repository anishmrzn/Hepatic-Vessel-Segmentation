import os
import torch

SEED = 42

DATA_DIR = "D:/monai-project/vessel_seg/data/Task08_HepaticVessel"
RUN_DIR = "D:/monai-project/vessel_seg/runs/vessel_segmentation_run_1"
CACHE_DIR = os.path.join(RUN_DIR, "dataset_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = (96, 96, 96)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

BATCH_SIZE = 1
NUM_WORKERS = 1

PIXDIM = (1.5, 1.5, 2.0)

A_MIN = -175.0
A_MAX = 250.0
B_MIN = 0.0
B_MAX = 1.0

MAX_EPOCHS = 30
VAL_INTERVAL = 5

LEARNING_RATE = 1e-4

KERNELS = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

INFERER_ROI_SIZE = PATCH_SIZE
INFERER_SW_BATCH_SIZE = 4
INFERER_OVERLAP = 0.5

MODEL_PATH = os.path.join(RUN_DIR, "best_metric_model.pth")