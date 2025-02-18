import torch
import torch.nn as nn
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.s = cfg_scale

    def forward(self, x, timesteps, cond=None, mask=None, cond_mask=None):
        B, T, D = x.shape

        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        if cond is not None:
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        if mask is not None:
            uncond_mask = mask - cond_mask
            mask = torch.cat([mask, uncond_mask], dim=0)

        if cond_mask is not None:
            cond_mask = torch.cat([cond_mask, cond_mask], dim=0)

        # uncond_mask = torch.cat([torch.ones_like(mask), torch.zeros_like(mask)], dim=0)

        out = self.model(x_combined, timesteps_combined, cond=cond, mask=mask, cond_mask=cond_mask)

        out_cond = out[:B]
        out_uncond = out[B:]

        cfg_out = self.s *  out_cond + (1-self.s) *out_uncond
        return cfg_out


class ClassCond_ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.s = cfg_scale

    def forward(self, x, timesteps, cond=None, mask=None, cond_mask=None, class_id=None, class_mask=None):
        B, T, D = x.shape

        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        if cond is not None:
            cond = torch.cat([cond, cond], dim=0)
        if mask is not None:
            mask = torch.cat([mask, mask], dim=0)
        if cond_mask is not None:
            cond_mask = torch.cat([cond_mask, cond_mask], dim=0)

        class_id = torch.cat([class_id, class_id], dim=0)
        class_mask = torch.cat([class_mask, torch.zeros_like(class_mask)], dim=0)

        out = self.model(x_combined, timesteps_combined, cond=cond, mask=mask, cond_mask=cond_mask, class_id=class_id, class_mask=class_mask)

        out_cond = out[:B]
        out_uncond = out[B:]

        cfg_out = self.s *  out_cond + (1-self.s) *out_uncond
        return cfg_out
