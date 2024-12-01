import torch
import os
from dataclasses import dataclass

@dataclass
class BaseConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET = "Cifar-10"  # "MNIST", "Cifar-10", "Cifar-100", "Flowers"

    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join("Logs_Checkpoints", "Inference")
    root_checkpoint_dir = os.path.join("Logs_Checkpoints", "checkpoints")

    # Current log and checkpoint directory.
    log_dir = "version_0"
    checkpoint_dir = "version_0"


@dataclass
class TrainingConfig:
    TIMESTEPS = 1000  # Define number of diffusion timesteps
    # IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32)
    IMG_SHAPE = (3, 32, 32)
    # NUM_EPOCHS = 800
    # BATCH_SIZE = 512
    LR = 2e-4 #2e-4
    # NUM_WORKERS = 0