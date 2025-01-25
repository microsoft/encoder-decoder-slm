import torch


class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_resampler(model_args, delay_load=False, **kwargs):
    resampler_type = getattr(model_args, "mm_resampler_type", None)

    return IdentityMap()
