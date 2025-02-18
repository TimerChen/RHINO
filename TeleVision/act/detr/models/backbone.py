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

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
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

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
        iphone: str = "none",
    ):
        if iphone in ["depth", "low_depth"]:
            # backbone = torchvision.models.resnet18(replace_stride_with_dilation=[False, False, dilation],)
            backbone = torchvision.models.resnet18(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=True,
            )
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(),
                norm_layer=FrozenBatchNorm2d,
            )  # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class DINOv2BackBone(nn.Module):
    def __init__(self, patch_h, patch_w) -> None:
        super().__init__()
        self.body = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        # self.body.eval()
        # for param in self.body.parameters():
        #     param.requires_grad = False
        self.num_channels = 384
        self.patch_h = patch_h
        self.patch_w = patch_w

    @torch.no_grad()
    def forward(self, tensor):
        xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
        od = OrderedDict()
        od["0"] = xs.reshape(xs.shape[0], self.patch_w, self.patch_h, 384).permute(0, 3, 2, 1)
        return od


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    # def forward (self, tensor):
    #     xs = self[0](tensor)
    #     pos = self[1](xs)
    #     return xs, pos

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args, iphone="none"):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.backbone == "dino_v2":
        backbone = DINOv2BackBone(args.patch_h, args.patch_w)
    else:
        assert args.backbone in ["resnet18", "resnet34"]
        backbone = Backbone(
            args.backbone, train_backbone, return_interm_layers, args.dilation, iphone
        )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
