# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        if name.startswith('resnet'):
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
            super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        elif name.startswith('clip'):
            import clip
            clip_model_name = name.split('_')[1]
            model, _ = clip.load(clip_model_name, jit=False)
            model = model.float()
            backbone = model.visual
            num_channels_dict = {
                'RN50': 2048,  # raw CLIP's layer 4 dimension is too high and cause OOM on 4090. Need to do some dimensionality reduction
            }
            num_channels = num_channels_dict[clip_model_name]
            super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        else:
            raise NotImplementedError(f"Backbone {name} not implemented")

class MaskclipBackbone(nn.Module):
    def __init__(self, backbone, train_backbone):
        super().__init__()
        import maskclip_onnx
        self.model, self.preprocess = maskclip_onnx.clip.load(
            "ViT-B/16",
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.patch_size = self.model.visual.patch_size
        self.num_channels = 512  # fixed with the model

    @torch.no_grad()
    def forward(self, img):
        b, _, input_size_h, input_size_w = img.shape 
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size 
        features = self.model.get_patch_encodings(img).to(torch.float32)
        clip_aligned_feats_bchw = features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)
        return {'0': clip_aligned_feats_bchw}

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        # For ResNet-18: xs := {'0': [B, 512, H // 32, W // 32]}
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    assert train_backbone is True
    return_interm_layers = args.masks
    if args.backbone.startswith('maskclip'):
        backbone = MaskclipBackbone(args.backbone, train_backbone)
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    model.patch_size = backbone.patch_size
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='maskclip', type=str, help='Name of the convolutional backbone to use')
    parser.add_argument('--dilation', action='store_true', help='If true, we replace stride with dilation in the last two blocks')
    parser.add_argument('--masks', action='store_true', help='If true, we output all the intermediate layers as a dict')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help='Learning rate for the backbone')
    args = parser.parse_args()
    model = build_backbone(args)
    img = torch.randn(1, 3, 224, 224)
    out = model(img)
    e()