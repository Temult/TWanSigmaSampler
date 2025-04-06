# __init__.py
from .TWanVideoSigmaSampler import TWanVideoSigmaSampler

NODE_CLASS_MAPPINGS = {
    "TWanVideoSigmaSampler": TWanVideoSigmaSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TWanVideoSigmaSampler": "WanVideo Sampler (Custom Sigma Input)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
