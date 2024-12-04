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
    NUM_CLASS = 10
    NUM_EPOCHS = 800
    BATCH_SIZE = 256
    LR = 2e-4 #2e-4
    GENERATE_VIDEO = True
    EXT = ".mp4" if GENERATE_VIDEO else ".png" 
    # NUM_WORKERS = 0
    CHECKPOINT_INTERVAL = 20  # 每10轮保存一次checkpoint
    MAX_CHECKPOINTS = 5      # 最多保留5个checkpoint


@dataclass
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 256
    BASE_CH_MULT = (1, 2, 4, 4) # 32, 16, 8, 8
    APPLY_ATTENTION = (False, True, True, False) #(False, True, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 4 # 128