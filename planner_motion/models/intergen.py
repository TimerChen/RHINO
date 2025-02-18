import torch
# import clip

from torch import nn
from models import *
from models import InterDiffusion, InterMotionGen


class InterGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.skip_text = cfg.SKIP_TEXT
        self.cond_class = cfg.get("COND_CLASS", 0)
        self.latent_dim = cfg.LATENT_DIM
        if cfg.DIFFUSION:
            self.decoder = InterDiffusion(cfg, sampling_strategy=cfg.STRATEGY)
        else:
            self.decoder = InterMotionGen(cfg)

        if not self.skip_text:
            # clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

            # self.token_embedding = clip_model.token_embedding
            # self.clip_transformer = clip_model.transformer
            # self.positional_embedding = clip_model.positional_embedding
            # self.ln_final = clip_model.ln_final
            # self.dtype = clip_model.dtype

            # set_requires_grad(self.clip_transformer, False)
            # set_requires_grad(self.token_embedding, False)
            # set_requires_grad(self.ln_final, False)
            clipTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation="gelu",
                batch_first=True)
            self.clipTransEncoder = nn.TransformerEncoder(
                clipTransEncoderLayer,
                num_layers=2)
            self.clip_ln = nn.LayerNorm(768)
            
    def set_normalizer(self, norm_kwargs):
        if self.cfg.DIFFUSION:
            return self.decoder.diffusion.set_normalizer(norm_kwargs)
        else:
            return self.decoder.set_normalizer(norm_kwargs)

    def compute_loss(self, batch):
        batch = self.text_process(batch)
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.decoder(batch))
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        batch = self.text_process(batch)
        batch.update(self.decode_motion(batch))
        return batch

    def text_process(self, batch):
        if self.skip_text:
            return batch
        
        device = next(self.clip_transformer.parameters()).device
        raw_text = batch["text"]

        with torch.no_grad():

            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_transformer(x)
            x = x.permute(1, 0, 2)
            clip_out = self.ln_final(x).type(self.dtype)

        out = self.clipTransEncoder(clip_out)
        out = self.clip_ln(out)

        cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        batch["cond"] = cond

        return batch
