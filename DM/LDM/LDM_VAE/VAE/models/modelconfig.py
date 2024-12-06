from dataclasses import dataclass

@dataclass
class ModelConfig:
    LATENT_CHANNEL: int = 6
    DOWNSAMPLE: int = 1
    