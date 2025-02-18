
import torch
import torchvision.models
import torchvision.transforms as transforms
# from torchvision.transforms import v2

from models.utils import *
from models.cfg_sampler import ClassCond_ClassifierFreeSampleModel
from models.blocks import *
from utils.utils import *

from models.gaussian_diffusion import (
    MotionDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
import random

class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.nfeats = cfg.HUMAN_BODY_DIM + cfg.HAND_DIM + 2 * cfg.NUM_OBJ
        self.nfeats_h1 = cfg.H1_BODY_DIM + cfg.HAND_DIM + 2 * cfg.NUM_OBJ
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.nfeats_h1, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)
        
    def forward(self, batch):
        x, seq_mask = batch["motions"], batch["seq_mask"]
        B, T, D  = x.shape
        x = x[:, :, self.nfeats:self.nfeats+self.nfeats_h1]
        x_emb = self.embed_motion(x)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][:,None], x_emb], dim=1)

        seq_mask = seq_mask[:, :, 1]  # only h1
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=-1).bool()  # (B, T+1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])

        batch["motion_emb"] = motion_emb

        return batch

class StopHERE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class InterDenoiser(nn.Module):
    def __init__(self,
                 input_feats,
                 input_h1_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 skip_text=False,
                 num_class=None,
                 backbone_type="cross",
                 pred_human=False,
                 **kargs):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.input_h1_feats = input_h1_feats
        self.time_embed_dim = latent_dim
        self.backbone_type = backbone_type
        self.pred_human = pred_human

        self.skip_text = skip_text
        self.num_class = 0 if num_class is None else num_class

        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.h1_embed = nn.Linear(self.input_h1_feats, self.latent_dim)
        self.class_embed = nn.Embedding(self.num_class, self.latent_dim)
        # self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
            
        # Output Module
        self.out = FinalLayer(self.latent_dim, self.input_feats, zero_init=False)
        self.out_h1 = FinalLayer(self.latent_dim, self.input_h1_feats, zero_init=False)

    def _forward_cross(self, embs, mask=None, cond=None, cond_mask=None, pos_shift=None):
        a_emb, b_emb, t_emb = embs
        B, T = a_emb.shape[0], a_emb.shape[1]

        h_a_prev = self.sequence_pos_encoder(a_emb, pos_shift)
        h_b_prev = self.sequence_pos_encoder(b_emb, pos_shift)

        if mask is None:
            mask = torch.ones(B, T).to(a_emb.device)
        
        key_padding_mask = ~(mask > 0.5)

        for i,block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev, t_emb, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, t_emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        output_a = self.out(h_a)
        output_b = self.out_h1(h_b)
        
        if self.skip_text:
            # replace cond with history mask
            output_a = output_a * (1-cond_mask) + cond[...,:self.input_feats] * cond_mask
            output_b = output_b * (1-cond_mask) + cond[...,self.input_feats:self.input_feats+self.input_h1_feats] * cond_mask

        output = torch.cat([output_a, output_b], dim=-1)
        
        print_check_tensor(output_a, "out a")
        print_check_tensor(output_b, "out b")

        return output

    def _forward_seq(self, embs, mask=None, cond=None, cond_mask=None, pos_shift=None):
        """
            stack two motion sequences
            seq = [(m1, m_h1), (m1, m_h1), (m1, m_h1)]
        """
        a_emb, b_emb, t_emb = embs
        B, T = a_emb.shape[0], a_emb.shape[1]

        h_a_prev = self.sequence_pos_encoder(a_emb, pos_shift)
        h_b_prev = self.sequence_pos_encoder(b_emb, pos_shift)

        if mask is None:
            mask = torch.ones(B, T).to(a_emb.device)
            
        # stack inputs
        # (B, 2, T, D) -> (B, T, 2, D)
        stacked_inputs = torch.stack((h_a_prev, h_b_prev), dim=1).permute(0, 2, 1, 3).reshape(B, 2*T, -1)
        # stacked_t_emb = torch.stack((t_emb, t_emb), dim=1).permute(0, 2, 1, 3).reshape(B, 2*T, -1)
        stacked_masks = torch.stack((mask, mask), dim=1).permute(0, 2, 1).reshape(B, 2*T)
        
        key_padding_mask = ~(stacked_masks > 0.5)
        h_ab_prev = stacked_inputs

        for i,block in enumerate(self.blocks):
            h_ab = block(h_ab_prev, h_ab_prev, t_emb, key_padding_mask)
            h_ab_prev = h_ab

        h_ab = h_ab.reshape(B, T, 2, -1)
        h_a = h_ab[..., 0, :]
        h_b = h_ab[..., 1, :]
        
        output_a = self.out(h_a)
        output_b = self.out_h1(h_b)
        
        if self.skip_text:
            # replace cond with history mask
            output_a = output_a * (1-cond_mask) + cond[...,:self.input_feats] * cond_mask
            output_b = output_b * (1-cond_mask) + cond[...,self.input_feats:self.input_feats+self.input_h1_feats] * cond_mask

        output = torch.cat([output_a, output_b], dim=-1)
        
        print_check_tensor(output_a, "out a")
        print_check_tensor(output_b, "out b")

        return output

        

    def forward(self, x, timesteps, mask=None, cond=None, cond_mask=None, 
                pos_shift=None, class_id=None, class_mask=None):
        """
        x: (B, T, feat_dim + h1_feat_dim)
        """
        B, T = x.shape[0], x.shape[1]

        if self.skip_text:
            # replace cond with history mask
            cond = cond.reshape(B, T, 2, -1)
            # ignore all rotation inputs of cond
            # cond[..., -13*6:] = 0.
            cond = cond.reshape(B, T, -1)

            # provide history input
            cond_mask = cond_mask.unsqueeze(-1)
            x = x * (1-cond_mask) + cond * cond_mask
            # do not predict human
            if not self.pred_human:
                x[...,:self.input_feats] = cond[...,:self.input_feats]

        x_a, x_b = x[...,:self.input_feats], x[...,self.input_feats:self.input_feats+self.input_h1_feats]

        if mask is not None:
            mask = mask[...,0]

        a_emb = self.motion_embed(x_a)
        b_emb = self.h1_embed(x_b)

        if not self.skip_text:
            emb = self.embed_timestep(timesteps) + self.text_embed(cond)
        elif self.num_class:
            emb = self.embed_timestep(timesteps) + self.class_embed(class_id) * class_mask
        else:
            emb = self.embed_timestep(timesteps)

        if self.backbone_type == "cross":
            output = self._forward_cross([a_emb, b_emb, emb], mask, cond, cond_mask, pos_shift)
        else:
            output = self._forward_seq([a_emb, b_emb, emb], mask, cond, cond_mask, pos_shift)

        return output

class InterMotionGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.HUMAN_BODY_DIM + cfg.HAND_DIM + 2 * cfg.NUM_OBJ
        self.nfeats_h1 = cfg.H1_BODY_DIM + cfg.HAND_DIM + 2 * cfg.NUM_OBJ
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP

        # for reaction model
        self.skip_text = cfg.SKIP_TEXT

        self.cfg_weight = cfg.CFG_WEIGHT
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER

        self.net = InterDenoiser(self.nfeats, self.nfeats_h1, self.latent_dim, ff_size=self.ff_size, num_layers=self.num_layers,
                                       num_heads=self.num_heads, dropout=self.dropout, activation=self.activation, 
                                       cfg_weight=self.cfg_weight, skip_text=self.skip_text,
                                       num_class=cfg.COND_CLASS, backbone_type=cfg.BACKBONE, pred_human=cfg.PRED_HUMAN)

    # def mask_cond(self, cond, cond_mask_prob = 0.1, force_mask=False):
    #     # TODO: add random mask for class_id
    #     bs = cond.shape[0]
    #     if force_mask:
    #         return torch.zeros_like(cond)
    #     elif cond_mask_prob > 0.:
    #         mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view([bs]+[1]*len(cond.shape[1:]))  # 1-> use null_cond, 0-> use real cond
    #         return cond * (1. - mask), (1. - mask)
    #     else:
    #         return cond, None
    
    def set_normalizer(self, norm_kwargs):
        self.normalizer = Normalizer(**norm_kwargs)
        return self.normalizer

    def generate_src_mask(self, T, seq_mask):
        B, T = seq_mask.shape[0], seq_mask.shape[1]
        src_mask = torch.ones(B, T, self.nfeats+self.nfeats_h1)
        src_mask[:, :, :self.nfeats] = seq_mask[:, :, 0].unsqueeze(-1).expand(-1, -1, self.nfeats)
        src_mask[:, :, self.nfeats:self.nfeats + self.nfeats_h1] = seq_mask[:, :, 1].unsqueeze(-1).expand(-1, -1, self.nfeats_h1)
        return src_mask

    def compute_loss(self, batch):
        cond = batch["cond"]
        cond_mask = batch.get("cond_mask", None)
        gt_motion = batch["motions"]  # gt motion
        
        gt_motion = self.normalizer.forward(gt_motion)
        cond = self.normalizer.forward(cond)
        
        seq_mask = batch.get("seq_mask", None)
        seq_mask = self.generate_src_mask(batch["motions"].shape[1], seq_mask).to(gt_motion.device)
        
        class_id = batch.get("label", None)
        class_mask = batch.get("class_mask", None)
        if class_mask is None:
            class_mask = torch.bernoulli(torch.ones(gt_motion.shape[0], device=cond.device)).reshape(-1, 1)
        
        x = torch.zeros_like(gt_motion)
        output = self.net(x, 0, cond=cond, cond_mask=cond_mask, class_id=class_id, class_mask=class_mask)
        gt_motion = self.normalizer.backward(gt_motion, )
        output = self.normalizer.backward(output, )
        
        loss_dict = dict()
        all_l1 = F.l1_loss(gt_motion, output, reduction="none")
        print("all_l1", all_l1.shape)
        print("seq_mask", seq_mask.shape)
        l1 = (all_l1 * seq_mask).mean()
        loss_dict["l1_loss"] = l1
        loss_dict["total"] = l1
        loss_dict["simple"] = l1
        loss_dict["H1"] = l1
        
        return loss_dict

    def forward(self, batch):
        cond = batch["cond"]
        cond_mask = batch["cond_mask"]
        x = torch.zeros_like(batch["motions"])
        
        output = self.net(x, 0, cond=cond, cond_mask=cond_mask, class_id=batch.get("class_id", None), class_mask=batch.get("class_mask", None))

        return {"output":output}


class InterDiffusion(nn.Module):
    def __init__(self, cfg, sampling_strategy="ddim50"):
        super().__init__()
        self.cfg = cfg
        # self.nfeats = cfg.INPUT_DIM
        # self.nfeats_h1 = cfg.INPUT_H1_DIM
        self.nfeats = cfg.HUMAN_BODY_DIM + cfg.HAND_DIM + 2 * cfg.NUM_OBJ
        self.nfeats_h1 = cfg.H1_BODY_DIM + cfg.HAND_DIM + 2 * cfg.NUM_OBJ
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP

        # for reaction model
        self.skip_text = cfg.SKIP_TEXT

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER
        self.sampling_strategy = sampling_strategy

        self.net = InterDenoiser(self.nfeats, self.nfeats_h1, self.latent_dim, ff_size=self.ff_size, num_layers=self.num_layers,
                                       num_heads=self.num_heads, dropout=self.dropout, activation=self.activation, 
                                       cfg_weight=self.cfg_weight, skip_text=self.skip_text,
                                       num_class=cfg.COND_CLASS, backbone_type=cfg.BACKBONE, pred_human=cfg.PRED_HUMAN)


        self.diffusion_steps = self.diffusion_steps
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing=[self.diffusion_steps]
        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            skip_text=self.skip_text,
            pred_human=cfg.PRED_HUMAN,
            vel_loss=cfg.VEL_LOSS,
        )
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

    def mask_cond(self, cond, cond_mask_prob = 0.1, force_mask=False):
        # TODO: add random mask for class_id
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view([bs]+[1]*len(cond.shape[1:]))  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask), (1. - mask)
        else:
            return cond, None

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask


    def compute_loss(self, batch):
        cond = batch["cond"]
        cond_mask = batch.get("cond_mask", None)
        x_start = batch["motions"]
        seq_mask = batch.get("seq_mask", None)
        class_id = batch.get("label", None)
        class_mask = batch.get("class_mask", None)
        
        B,T = batch["motions"].shape[:2]
        if class_mask is None:
            class_mask = torch.bernoulli(torch.ones(x_start.shape[0], device=cond.device)).reshape(-1, 1)
        # if cond is not None:
        #     prob = 0 if self.skip_text else 0.1
        #     if cond_mask is None:
        #         cond, cond_mask = self.mask_cond(cond, prob)

        if seq_mask is None:
            seq_mask = self.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"]).to(x_start.device)

        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=seq_mask,
            t_bar=self.cfg.T_BAR,
            cond_mask=cond_mask,
            model_kwargs={"mask":seq_mask,
                          "cond":cond,
                          "cond_mask":cond_mask,
                          "class_id": class_id,
                          "class_mask": class_mask,
                          },
        )
        return output

    def forward(self, batch):
        cond = batch["cond"]
        cond_mask = batch["cond_mask"]
        # x_start = batch["motions"]
        B = cond.shape[0]
        T = cond.shape[1]

        timestep_respacing= self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            skip_text=self.skip_text,
            pred_human=self.cfg.PRED_HUMAN,
            vel_loss=self.cfg.VEL_LOSS,
        )

        # disable cfg model
        self.cfg_model = ClassCond_ClassifierFreeSampleModel(self.net, self.cfg_weight)
        output = self.diffusion_test.ddim_sample_loop(
            self.net,
            (B, T, self.nfeats+self.nfeats_h1),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "mask":None, # check mask???
                "cond":cond,
                "cond_mask":cond_mask,
                "class_id": batch.get("class_id", None),
                "class_mask": batch.get("class_mask", None)
            },
            x_start=None)
        return {"output":output}



class HumanClassifierNet(nn.Module):
    def __init__(self,
                 input_feats,
                 num_class,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 skip_text=False,
                 hand_input_dim=6,
                 **kargs):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.num_class = num_class

        self.skip_text = skip_text

        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        # self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
        
        # Output Module
        self.final = FinalLayer(self.latent_dim, self.input_feats)
        # image augment
        trans = [
            transforms.RandomHorizontalFlip(p=0.5), # randomly flip and rotate
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(0.3,0.4,0.4,0.2),   
        ]
        trans = [torchvision.transforms.Resize((128, 128))] + trans\
            + [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        self.img_trans = torchvision.transforms.Compose(trans)
        self.hand_resnet = torchvision.models.resnet18(replace_stride_with_dilation=[False, False, False],pretrained=True)
        # nn.Linear(512 * block.expansion, num_classes)
        self.hand_resnet.fc = nn.Linear(512, self.latent_dim)
        self.hand_img_merge = nn.Linear(self.latent_dim*2, self.latent_dim)
        
        self.hand_input_emb = nn.Linear(hand_input_dim, self.latent_dim)
        # trans = [v2.Resize((self.patch_h * self.patch_size, self.patch_w * self.patch_size)),
        # trans = [v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        # self.img_trans = v2.Compose(trans)
        
        self.classifier = MLPClassifier(self.input_feats, self.num_class)
        

    def forward(self, x, mask=None, cond=None, 
                cond_mask=None, pos_shift=None,
                hand_pos=None, hand_img=None, hand_near=None, head_pos=None):
        """
        x: B, T, D
        """
        timesteps=torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        B, T = x.shape[0], x.shape[1]


        # x_a, x_b = x[...,:self.input_feats], x[...,self.input_feats:]
        x_a = x[...,:self.input_feats]
        a_emb = self.motion_embed(x_a)

        if mask is not None:
            mask = mask[...,0]
        hand_input = []
        
        if hand_pos is not None:
            hand_input.append(hand_pos)
        if hand_near is not None:
            hand_input.append(hand_near)
        if head_pos is not None:
            hand_input.append(head_pos)
            
        # if hand_pos is not None:
        if len(hand_input) > 0:
            # (B, 4) -> (B, 1, D)
            hand_input = torch.cat(hand_input, dim=-1)
            assert torch.isnan(hand_input).sum() == 0, f"hand_pos has nan: {hand_input}"
            hand_input = self.hand_input_emb(hand_input).unsqueeze(1)
            a_emb = a_emb + hand_input
            
        if hand_img is not None:
            hand_img = hand_img.reshape((hand_img.shape[0]*hand_img.shape[1],)+hand_img.shape[2:])
            hand_img = self.img_trans(hand_img)
            hand_img = self.hand_resnet(hand_img)
            hand_img = self.hand_img_merge(hand_img.reshape((B, -1))).unsqueeze(1)
            a_emb = a_emb + hand_img
                    
        # b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb, pos_shift)
        # h_b_prev = self.sequence_pos_encoder(b_emb, pos_shift)

        if not self.skip_text:
            emb = self.embed_timestep(timesteps) + self.text_embed(cond)
        else:
            emb = self.embed_timestep(timesteps)

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        
        key_padding_mask = ~(mask > 0.5)

        for i,block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_a_prev, emb, key_padding_mask)
            # h_b = block(h_b_prev, h_a_prev, emb, key_padding_mask)
            h_a_prev = h_a
            # h_b_prev = h_b

        output_a = self.final(h_a)
        # output_b = self.out(h_b
        
        output = self.classifier(output_a[:, -1])
        return output

class HumanClassifier(nn.Module):
    def __init__(self,
                 input_feats,
                 num_class,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 skip_text=False,
                 hd_input=[],
                 hand_input_dim=6,
                 **kargs):
        super().__init__()
        self.num_class = num_class
        self.hd_input = hd_input
        self.net = HumanClassifierNet(input_feats, num_class, latent_dim, num_frames, 
                                             ff_size, num_layers, num_heads, dropout, 
                                             activation, cfg_weight, skip_text, hand_input_dim=hand_input_dim)
        
    def set_normalizer(self, norm_kwargs):
        self.normalizer = Normalizer(**norm_kwargs)
        return self.normalizer
        
    def forward(self, batch):
        return self.compute_loss(batch)
    
    def normalize(self, batch):
        # motion = batch["motions"]
        # hand_pos = batch["hand_pos"]
        # hand_near = batch["hand_near"]
        # print("111motion mean, max, min", motion.mean(), motion.max(), motion.min())
        # print("111hand_pos mean, max, min", hand_pos.mean(), hand_pos.max(), hand_pos.min())
        # print("111hand_near mean, max, min", hand_near.mean(), hand_near.max(), hand_near.min())
        hand_pos = batch.get("hand_pos", None) if "hand_pos" in self.hd_input else None
        hand_near = batch.get("hand_near", None) if "hand_near" in self.hd_input else None
        hand_imgs = batch.get("hand_imgs", None) if "hand_imgs" in self.hd_input else None
        head_pos = batch.get("head_pos", None) if "head_pos" in self.hd_input else None
        motion = self.normalizer.forward(batch["motions"])
        hand_pos = self.normalizer.forward(hand_pos, name="hand_pos") if hand_pos is not None else None
        hand_near = self.normalizer.forward(hand_near, name="hand_near") if hand_near is not None else None
        head_pos = self.normalizer.forward(head_pos, name="head_pos") if head_pos is not None else None
        # print("motion mean, max, min", motion.mean(), motion.max(), motion.min())
        # print("hand_pos mean, max, min", hand_pos.mean(), hand_pos.max(), hand_pos.min())
        # print("hand_near mean, max, min", hand_near.mean(), hand_near.max(), hand_near.min())
        return motion, hand_pos, hand_near, hand_imgs, head_pos
    
    def infer(self, batch):
        motion, hand_pos, hand_near, hand_imgs, head_pos = self.normalize(batch)
        return self.net(motion, batch["mask"], 
                        hand_pos=hand_pos, 
                        hand_img=hand_imgs,
                        hand_near=hand_near,
                        head_pos=head_pos)
    
    def compute_loss(self, batch):
        # criterion = nn.NLLLoss() # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        motion, hand_pos, hand_near, hand_imgs, head_pos = self.normalize(batch)
        
        outputs = self.net(motion, batch["mask"], 
                           hand_pos=hand_pos, 
                           hand_img=hand_imgs,
                           hand_near=hand_near,
                           head_pos=head_pos)
        # print(outputs.argmax(dim=1))
        # exit()
        loss = criterion(outputs, batch["labels"])
        accu = (outputs.argmax(dim=1) == batch["labels"]).float().mean()
        loss_info = {
            "total": loss,
            "accu": accu,
            "correct_no_idle": ((outputs.argmax(dim=1) == batch["labels"]) & (batch["labels"] > 0)).float().sum(),
            "cnt_no_idle": (batch["labels"] > 0).float().sum(),
        }
        for i in range(self.num_class):
            num0 = (outputs.argmax(dim=1) == i).float().mean()      
            loss_info[f"classified_num/{i}"] = num0
            
        label_pred = torch.concatenate([batch["labels"].reshape(-1,1), outputs.argmax(dim=1).reshape(-1,1)], dim=-1)
        loss_info["label_pred"] = label_pred
        return loss, loss_info


