import torch
import torch.nn.functional as F

from .model_v1 import BaseNetV1


class BaseNetHNContrast(BaseNetV1):
    """V1 inference graph with optional training-only multi-scale features."""

    def forward(self, x1, x2, return_features=False):
        features_pre = self.backbone(x1)
        features_post = self.backbone(x2)
        logits = self.swa(*(features_pre + features_post))
        target_size = x1.shape[-2:]
        outputs = tuple(
            F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            for x in logits
        )
        if return_features:
            return outputs, (features_pre, features_post)
        return outputs
