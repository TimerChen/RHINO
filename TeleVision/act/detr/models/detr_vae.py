# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .maskclip_backbone import build_backbone as build_maskclip_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np
import time

import IPython

e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        action_dim,
        num_queries,
        camera_names,
        patch_shape=None,
        ignore_image=False,
        iphone="none",
        ignore_rgb=False,
        num_action_heads=1, # Added parameter for number of action heads
        num_src_heads=1, # Added parameter for number of src embedding heads
    ):
        """Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            num_action_heads: number of action output heads
            num_src_heads: number of src embedding heads before transformer
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.patch_shape = patch_shape
        self.ignore_image = ignore_image
        self.ignore_rgb = ignore_rgb
        self.iphone = iphone
        self.num_action_heads = num_action_heads
        self.num_src_heads = num_src_heads
        hidden_dim = transformer.d_model
        
        # Create multiple action heads
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(num_action_heads)
        ])
        
        # Create multiple src embedding heads
        self.src_heads = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_src_heads)
        ])
        
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            raise NotImplementedError
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            action_dim, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(
            state_dim, hidden_dim
        )  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        def show(name, x):
            print(f"name {name}, ", end="")
            if not isinstance(x, list):
                print(f"{x.shape}, {torch.isnan(x).any()} {torch.max(x.abs())}")
                if torch.isnan(x).any():
                    exit()
            else:
                print("list ", len(x))
                for i, xx in enumerate(x):
                    show(f"-{i}", xx)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            if self.iphone != "none":
                if self.iphone in ["rgbd", "rgb_lowd"]:
                    depth = image[..., 3:, :, :].flatten(0, 1)
                    d_featuress, d_poss = self.backbones[1](depth)  # HARDCODED
                # split img and depth
                image = image[..., :3, :, :]
            
            img = image.flatten(0, 1)
            
            
            featuress, poss = self.backbones[0](img)  # HARDCODED
            camera_names = self.camera_names
            if self.iphone != "none":
                if self.iphone in ["rgbd", "rgb_lowd"]:
                    if self.ignore_image and not self.ignore_rgb:
                        featuress[0] = featuress[0][:, None] 
                        camera_names = ["rgb"]
                    elif self.ignore_rgb:
                        featuress = d_featuress
                        # add a dimension: [45, 512, 7, 10] -> [45, 1, 512, 7, 10]
                        featuress[0] = featuress[0][:, None] 
                        poss = d_poss
                        camera_names = ["depth"]
                    else:
                    # featuress[0] = featuress[0] + d_featuress[0]
                        # add a dimension: [45, 512, 7, 10] -> [45, 2, 512, 7, 10]
                        featuress[0] = torch.stack([featuress[0], d_featuress[0]], dim=1)
                        poss = poss + d_poss
                
                featuress = featuress[0]
                pos = poss[0]
            else:
                if self.patch_shape is not None:
                    patch_h, patch_w = self.patch_shape
                else:
                    patch_h = img.shape[2] // self.backbones[0].patch_size
                    patch_w = img.shape[3] // self.backbones[0].patch_size 
                if self.ignore_image:
                    featuress = [torch.zeros_like(f) for f in featuress]
                    # poss = [torch.zeros_like(p) for p in poss]
                    
                featuress = featuress[0].view(
                    image.shape[0], 2, self.backbones[0].num_channels, patch_h, patch_w
                ) 
                pos = poss[0]
            for cam_id, cam_name in enumerate(camera_names):
                # start = time.time()
                # import ipdb; ipdb.set_trace()
                features = featuress[:, cam_id]  # HARDCODED
                # features, pos = self.backbones[cam_id](image[:, cam_id]) # HARDCODED
                # print("Time for 1 backbone: ", time.time() - start, image.shape)
                # features = features[0] # take the last layer feature
                # pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos / 2 + cam_id - 0.5)
                # break

            # for cam_id, cam_name in enumerate(self.camera_names):
            #     features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
            #     features = features[0] # take the last layer feature
            #     pos = pos[0]
            #     all_cam_features.append(self.input_proj(features))
            #     all_cam_pos.append(pos)

            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            
            if self.ignore_image:
                # when ignore image, only keep one patch
                # batch, channel, height, width
                src = src[:, :, :1, :1]
                pos = pos[:, :, :1, :1]
            
            # Apply multiple src embedding heads before transformer
            if self.num_src_heads > 1:
                # Reshape src to apply linear layers
                b, c, h, w = src.shape
                src = src.flatten(2).transpose(1, 2)  # (b, h*w, c)
                # Apply each head and weight by qpos selector
                src_embeds = [head(src) for head in self.src_heads]
                src_embeds = torch.stack(src_embeds, dim=1)  # (batch, num_heads, h*w, c)
                # Use qpos selector for src heads
                src_selector = qpos[:, -self.num_src_heads:]
                src_selector = src_selector.unsqueeze(-1).unsqueeze(-1)  # [B, numhead, 1, 1]
                # Weighted sum across heads
                src = (src_embeds * src_selector).sum(dim=1)  # [B, h*w, c]
                # Reshape back
                src = src.transpose(1, 2).reshape(b, c, h, w)
            else:
                # Apply single head if num_src_heads=1
                b, c, h, w = src.shape
                src = src.flatten(2).transpose(1, 2)
                src = self.src_heads[0](src)
                src = src.transpose(1, 2).reshape(b, c, h, w)
                
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            raise NotImplementedError
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]
            
        # Get outputs from all action heads
        a_hats = [head(hs) for head in self.action_heads]
        # Select output based on head_selector if provided
        if self.num_action_heads > 1:
            # Stack all head outputs: (batch, num_heads, H, W)
            a_hats = torch.stack(a_hats, dim=1)
            # Use head_selector to select appropriate output
            # head_selector: (batch, num_heads)
            # Expand head_selector to match a_hats dimensions
            head_selector = qpos[:, -self.num_action_heads:]
            head_selector = head_selector.unsqueeze(-1).unsqueeze(-1)  # [B, numhead, 1, 1]
            # Weighted sum across heads (one-hot selector means only one head contributes)
            a_hat = (a_hats * head_selector).sum(dim=1)  # [B, H, W]
        else:
            # Default to first head if no selector provided
            a_hat = a_hats[0]
            
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim)  # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5),
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(
                input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2
            )
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            pos = pos[0]  # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)  # 768 each
        features = torch.cat([flattened_features, qpos], axis=1)  # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = args.state_dim
    action_dim = args.action_dim

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    # backbone = build_backbone(args)
    # backbones.append(backbone)
    # for _ in args.camera_names:
    if args.iphone != "none":
        patch_shape = (args.patch_h, args.patch_w)
        backbone = build_backbone(args)
        if args.iphone in ["rgbd", "rgb_lowd"]:
            depth_backbone = build_backbone(args,iphone="depth")
            backbones = [backbone, depth_backbone]
        else:
            backbones = [backbone]
    else:
        if args.backbone.startswith('maskclip'):
            backbone = build_maskclip_backbone(args)
            patch_shape = None
        else:
            backbone = build_backbone(args)
            patch_shape = (args.patch_h, args.patch_w)
        backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)


    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        action_dim=action_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        patch_shape=patch_shape,
        ignore_image=args.ignore_image,
        iphone = args.iphone,
        ignore_rgb=args.ignore_rgb,
        num_action_heads=args.num_action_heads,
        num_src_heads=args.num_src_heads,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_cnnmlp(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model
