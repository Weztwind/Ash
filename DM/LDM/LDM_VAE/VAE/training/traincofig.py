import torch
from dataclasses import dataclass

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_PATH: str = "./results"
    DISCFACTOR: float = 1.0
    DISC_START: int = 1
    LR: float = 1e-4
    EPS: float = 1e-8
    BETA1: float = 0.5
    BETA2: float = 0.9
    NUM_EPOCHS: int = 400
    BATCH_SIZE: int = 128
