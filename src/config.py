EXPERIMENT_NAME = "mps_bn_extra_conv_seed44_60ep"

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 60
NUM_CLASSES = 10
SPLIT_SEED = 42  # Keep fixed across all runs
SEED = 44       # Training seed: 42, 43, or 44

USE_AUGMENTATION = True
WEIGHT_DECAY = 0.0

USE_LR_SCHEDULER = True
LR_SCHEDULER_MODE = "min"
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
MIN_LR = 1e-5

# Enable batch normalization after each convolution.
USE_BATCH_NORM = True

# Add a third convolution before the second pooling operation.
USE_EXTRA_CONV = True

# Prefer CUDA, then MPS, then CPU; set explicitly to require a device.
DEVICE = "auto"

# Number of training data workers used with CUDA.
CUDA_NUM_WORKERS = 2



# Resume from the last complete epoch of the same experiment.
RESUME_TRAINING = False