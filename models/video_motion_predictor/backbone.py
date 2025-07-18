import os
import json

import torch
import torch.nn as nn

from .modeling_config import VideoMAEv2Config
from .modeling_videomaev2 import VideoMAEv2

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
CONFIGS = {
    "base": os.path.abspath(os.path.join(CONFIGS_DIR, "base.json")),
}


def get_videomae2(
    config_path=None, model_type='base',
    num_classes=0, use_checkpoint=False
):
    if config_path is None:
        config_path = CONFIGS[model_type]

    # load config
    with open(config_path, 'r') as c_fd:
        config = json.load(c_fd)
    config = VideoMAEv2Config(**config)
    config.model_config['with_cp'] = bool(use_checkpoint)
    config.model_config['num_classes'] = num_classes

    # initialize model
    model = VideoMAEv2(config=config)

    return model, config


class VideoMaeBackbone(nn.Module):
    def __init__(
        self,
        model_type="base",
        output_dim=3,
        with_classifier=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.output_dim = (
            output_dim if with_classifier else
            None
        )
        self.use_checkpoint = use_checkpoint

        # videomae-v2
        self.model, self.config = get_videomae2(
            model_type=model_type,
            num_classes=self.output_dim if with_classifier else 0,
            use_checkpoint=bool(use_checkpoint)
        )
        self.embed_dim = self.config.model_config['embed_dim']
        self.num_frames = self.config.model_config['num_frames']

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.register_buffer(
            'image_mean', torch.tensor(image_mean).reshape(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            'image_std', torch.tensor(image_std).reshape(1, 1, 3, 1, 1)
        )

    def forward(self, frames):
        # pixel values should be correctly normalized
        frames = (frames - self.image_mean) / self.image_std

        # (b, f, c, h, w) --> (b, c, f, h, w)
        frames = frames.permute(0, 2, 1, 3, 4)

        feat = self.model(frames)
        return feat
