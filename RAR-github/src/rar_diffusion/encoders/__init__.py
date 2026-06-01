from rar_diffusion.encoders.clip import ClipEncoder
from rar_diffusion.encoders.dinov2 import Dinov2Encoder
from rar_diffusion.encoders.resnet import ResNetEncoder

ENCODERS = {
    "clip": ClipEncoder,
    "dinov2": Dinov2Encoder,
    "dino": Dinov2Encoder,
    "resnet": ResNetEncoder,
}


def build_encoder(encoder_type: str, encfg: dict, device: str):
    key = encoder_type.lower()
    if key not in ENCODERS:
        raise ValueError(f"Unknown encoder type '{encoder_type}'. Choose from: {sorted(ENCODERS)}")
    return ENCODERS[key](encfg, device)


__all__ = ["ENCODERS", "build_encoder", "ClipEncoder", "Dinov2Encoder", "ResNetEncoder"]
