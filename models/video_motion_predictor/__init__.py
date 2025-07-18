import os
import yaml
import torch
from easydict import EasyDict

from .model import VideoMotionPredictor


def get_motion_predictor(config=None, ckpt_path=None, load_ckpt=True, eval_mode=False):
    if config is None and ckpt_path is None:
        raise RuntimeError(
            "specify either config or checkpoint path with config"
        )

    elif config is None or config == '':
        ckpt_dir = os.path.dirname(ckpt_path)
        config_path = os.path.join(ckpt_dir, 'config.yaml')
        with open(config_path) as f:
            config = EasyDict(yaml.safe_load(f))

    if eval_mode:
        config.use_checkpoint = False

    model = VideoMotionPredictor(
        num_frames=config.num_frames,
        output_dim=len(config.targets),
        use_checkpoint=config.use_checkpoint,
        use_sigmoid=config.use_sigmoid,
        drop_prob=getattr(config, 'drop_prob', 0.0),
        targets_norm=getattr(config, 'targets_norm', [3, 3, 3])
    )

    if ckpt_path is not None and load_ckpt:
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        keys = list(state_dict.keys())
        for k in keys:
            new_k = k.replace('_orig_mod.', '')
            state_dict[new_k] = state_dict.pop(k)
            if new_k in ['mean', 'std']:
                state_dict[f'image_{new_k}'] = state_dict.pop(new_k)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded: {msg}")

    return model
