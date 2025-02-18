# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython

e = IPython.embed


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)  # will be overridden
    parser.add_argument("--lr_backbone", default=1e-5, type=float)  # will be overridden
    parser.add_argument("--batch_size", default=2, type=int)  # not used
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)  # not used
    parser.add_argument("--lr_drop", default=200, type=int)  # not used
    parser.add_argument(
        "--clip_max_norm",
        default=0.1,
        type=float,  # not used
        help="gradient clipping max norm",
    )
    parser.add_argument(
        "--qpos_noise_std",
        action="store",
        default=0,
        type=float,
        help="lr",
        required=False,
    )

    # Model parameters
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet18",
        type=str,  # will be overridden
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--camera_names",
        default=[],
        type=list,  # will be overridden
        help="A list of camera names",
    )
    parser.add_argument(
        "--patch_h",
        default=16,
        type=int, 
        help="patch height",
    )
    parser.add_argument(
        "--patch_w",
        default=22,
        type=int,
        help="patch width",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=4,
        type=int,  # will be overridden
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,  # will be overridden
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,  # will be overridden
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,  # will be overridden
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,  # will be overridden
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries",
        default=400,
        type=int,  # will be overridden
        help="Number of query slots",
    )
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    parser.add_argument("--ignore_image", action="store_true")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument("--state_name", nargs="+", type=str, default=["head", "arm", "wrist", "hand", "force"])
    parser.add_argument("--left_right_mask", type=str, default="11")
    parser.add_argument("--chop_data", type=int, default=None, nargs="+", )
    # parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--progress_bar", type=int, default=-1)
    parser.add_argument("--history_stack", type=int, default=0)
    parser.add_argument("--iphone", choices=["rgb", "rgbd", "rgb_lowd", "none"], default="none")
    parser.add_argument("--depth_type", type=str, default="depth", choices=["depth", "depth_any"])
    parser.add_argument("--ignore_rgb", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    # parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    # parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", required=True
    )
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument("--temporal_agg", action="store_true")

    parser.add_argument("--save_jit", action="store_true")
    parser.add_argument(
        "--ckpt_dir",
        default="/home/cxx/h1_hardware/data/logs",
        type=str,  # will be overridden
        help="ckpt_dir",
    )

    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--resumeid", action="store", type=str, help="resume id", required=False
    )
    parser.add_argument(
        "--resume_ckpt", action="store", type=str, help="resume ckpt", required=False
    )
    parser.add_argument(
        "--taskid", action="store", type=str, help="task id", required=True
    )
    parser.add_argument(
        "--exptid", action="store", type=str, help="experiment id", required=True
    )
    parser.add_argument("--source", choices=["self", "ssd"], default="self")
    
    parser.add_argument("--num_action_heads", type=int, default=1)
    parser.add_argument("--num_src_heads", type=int, default=1)
    parser.add_argument("--cancel_unsafe", action="store_true", default=False)

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    return model, optimizer
