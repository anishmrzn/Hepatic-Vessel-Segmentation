import os

DATA_DIR = 'D:/monai-project/vessel_seg/data/Task08_HepaticVessel/'

RAW_TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'imagesTr/')
RAW_TRAIN_LABELS_DIR = os.path.join(DATA_DIR, 'labelsTr/')

LIVER_ROI_MASKS_DIR = os.path.join(DATA_DIR, 'liver_roi_masks/')
os.makedirs(LIVER_ROI_MASKS_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

ISOTROPIC_RESOLUTION = (1.5, 1.5, 1.5)
PATCH_SIZE = (128, 128)

FRANGI_SCALES = (4, 8)
FRANGI_ALPHA = 0.5
FRANGI_BETA = 0.5
FRANGI_BLACK_RIDGES = True

CED_ITERATIONS = 10
CED_K_PARAM = 0.03
CED_LAMBDA_PARAM = 0.05
CED_OPTION = 1

BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

IN_CHANNELS = 1
OUT_CHANNELS = 1

SAVE_DIR = os.path.join(os.getcwd(), 'runs')
os.makedirs(SAVE_DIR, exist_ok=True)