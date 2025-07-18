from typing import List
import numpy as np
import imageio.v3 as iio

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .backbone import VideoMaeBackbone
from .transformer_head import TransformerHead


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples, device=x.device)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)


class VideoMotionPredictor(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_frames: int = 16,
        output_dim: int = 3,
        use_checkpoint: bool = False,
        use_sigmoid: bool = False,
        drop_prob: float = 0.0,
        targets_norm: List[float] = [3, 3, 3],
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_frames = num_frames
        self.use_sigmoid = use_sigmoid

        self.backbone = VideoMaeBackbone(
            model_type="base",
            output_dim=output_dim,
            with_classifier=False,
            use_checkpoint=use_checkpoint,
        )
        self.pad_token = nn.Parameter(
            torch.randn((1, self.backbone.embed_dim), dtype=torch.float32)
        )
        self.head = TransformerHead(
            self.backbone.embed_dim,
            output_dim,
            hidden_size=self.backbone.embed_dim,
            max_seq_len=240,
            depth=4,
            num_heads=8,
            use_cond=False,
            head_type="mlp",
            train_pos_embed=False,
            drop_prob=drop_prob
        )
        self.frame_size = self.backbone.config.model_config["img_size"]
        self.register_buffer(
            'targets_norm',
            torch.tensor(targets_norm, dtype=torch.float32).unsqueeze(0)
        )

    def set_backbone_mode(self, trainable: bool):
        self.backbone.requires_grad_(trainable)

    def forward_encoder(self, frame_seg, max_size=16):
        if max_size <= 0:
            max_size = frame_seg.shape[0]

        feat_list = []
        for f_chunk in frame_seg.split(max_size, dim=0):
            # feat: [B, feat_dim]
            feat = self.backbone(f_chunk)
            feat_list.append(feat)

        feat = torch.cat(feat_list, dim=0)
        return feat

    @property
    def device(self):
        return self.targets_norm.device

    def split_and_pad(self, feat_seq, num_seg):
        b = num_seg.shape[0]
        max_len = num_seg.max().item()
        idx = torch.cumsum(
            torch.cat([torch.tensor([0]).to(num_seg.device), num_seg]),
            dim=0
        )

        feat_list = []
        for i in range(b):
            s, e = idx[i], idx[i+1]
            pad_sz = max_len - (e - s)
            feat = torch.cat(
                [
                    feat_seq[s:e],
                    self.pad_token.expand(pad_sz, -1),
                ], dim=0
            )
            feat_list.append(feat)

        feat_seq = torch.stack(feat_list, dim=0)
        return feat_seq

    def forward_head(self, feat_seq, num_seg):
        feat = self.split_and_pad(feat_seq, num_seg)
        pred_reg = self.head(feat)
        return pred_reg

    def forward(self, frames, lens):
        '''
            frames: tensor of shape [sum(b_i * f_i), c, h, w] with pixels
                in range [0, 1]
            lens: tensor of sequence lengths for each element of
                the concatenated batch [b,]
        '''
        # pixel values should be correctly normalized
        _, c, h, w = frames.shape

        # reshape the input for processing with video encoder
        num_seg = lens // self.num_frames
        frames = frames.reshape(-1, self.num_frames, c, h, w)

        feat_pool = self.forward_encoder(frames)
        pred_reg = self.forward_head(feat_pool, num_seg)

        if self.use_sigmoid:
            pred_reg = torch.sigmoid(pred_reg)
            pred_reg = pred_reg * self.targets_norm

        return pred_reg

    def preprocess_data(self, video, fps: float = 4.0, max_frames: int = 240):
        frames = iio.imread(video, plugin="pyav")
        frames = np.stack(frames) # type: ignore
        frames = torch.from_numpy(frames).permute([0, 3, 1, 2])

        # resample frames with a fixed time between the frames
        meta = iio.immeta(video, plugin="pyav")
        num_frames = min(fps * int(meta['duration']), max_frames)

        # align to the size required by the encoder
        num_frames = int((num_frames // self.num_frames + 1) * self.num_frames)
        lens = torch.tensor([num_frames], dtype=torch.long)

        # sample frames from a video clip uniformly
        frames = uniform_temporal_subsample(
            frames, num_samples=num_frames,
            temporal_dim=0
        )

        # [0, 255] --> [0, 1]
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0

        # resize frames
        frames = torch.nn.functional.interpolate(
            frames, size=(self.frame_size, self.frame_size),
            mode='bilinear'
        )
        return frames, lens

    @torch.inference_mode()
    def inference(self, videos, fps: float = 4, max_frames: int = 240, return_dict=True):
        if not isinstance(videos, (list, tuple)):
            videos = [videos]

        frames_batch, lens_batch = [], []
        for v in videos:
            frames, lens = self.preprocess_data(v, fps, max_frames)
            frames_batch.append(frames)
            lens_batch.append(lens)

        frames_batch = torch.cat(frames_batch, dim=0)
        lens_batch = torch.cat(lens_batch, dim=0)

        outputs = self.forward(
            frames_batch.to(self.device),
            lens_batch.to(self.device)
        )
        if return_dict:
            cam, obj, dyn = outputs.unbind(dim=1)
            outputs = {
                "camera_movement_score": cam.cpu().tolist(),
                "object_movement_score": obj.cpu().tolist(),
                "dynamics_score": dyn.cpu().tolist(),
            }
        return outputs
