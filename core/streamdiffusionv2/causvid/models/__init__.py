"""Minimal model registry for StreamDiffusionV2 core-only runtime."""

from .wan.wan_wrapper import (
    CausalWanDiffusionWrapper,
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)

DIFFUSION_NAME_TO_CLASS = {
    "wan": WanDiffusionWrapper,
    "causal_wan": CausalWanDiffusionWrapper,
}

TEXTENCODER_NAME_TO_CLASS = {
    "wan": WanTextEncoder,
    "causal_wan": WanTextEncoder,
}

VAE_NAME_TO_CLASS = {
    "wan": WanVAEWrapper,
    "causal_wan": WanVAEWrapper,
}


def get_diffusion_wrapper(model_name: str):
    return DIFFUSION_NAME_TO_CLASS[model_name]


def get_text_encoder_wrapper(model_name: str):
    return TEXTENCODER_NAME_TO_CLASS[model_name]


def get_vae_wrapper(model_name: str):
    return VAE_NAME_TO_CLASS[model_name]
