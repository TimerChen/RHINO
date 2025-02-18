import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import v2
import torch

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython

e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.iphone = args_override["iphone"]
        self.kl_weight = args_override["kl_weight"]
        self.qpos_noise_std = args_override["qpos_noise_std"]
        self.backbone = args_override["backbone"]
        if args_override["backbone"] == "maskclip":
            print("Using MaskClip backbone")
            self.patch_size = 16
        else:
            self.patch_size = 14
        self.patch_h = args_override["patch_h"]
        self.patch_w = args_override["patch_w"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        if actions is not None:
            if self.backbone in ["maskclip", "dino_v2"]:
                transform_ops = [v2.ColorJitter(
                                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                                ),
                                v2.RandomPerspective(distortion_scale=0.5),
                                v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)),]
                
                # 68.27% 95.45% 99.7% in 1 sigma 2 sigma 3 sigma
                # here, let noise < 0.1, then sigma = 0.1/3
                # qpos_noise_std = (0.1/3) ** 2 = 0.00111...
                qpos += (self.qpos_noise_std**0.5) * torch.randn_like(qpos)
            else:
                transform_ops = []
        else:
            transform_ops = []

        if self.backbone != "maskclip": # dino_v2, resnet18, resnet34
            transform_ops += [v2.Resize((self.patch_h * self.patch_size, self.patch_w * self.patch_size)),
                              v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        else:
            transform_ops += [v2.Resize((self.patch_h * self.patch_size, self.patch_w * self.patch_size)),
                              v2.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
            
        transform = v2.Compose(transform_ops)

        if self.iphone != "none":
            if self.iphone in ["rgbd", "rgb_lowd"]:
                depth = image[..., 3:, :, :]
                image = image[..., :3, :, :]
            else:
                image = image[..., :3, :, :]
            image = transform(image)
            if self.iphone in ["rgbd", "rgb_lowd"]:
                dtransform = v2.Compose([v2.Resize((self.patch_h * self.patch_size, self.patch_w * self.patch_size)),])
                depth = dtransform(depth)
                image = torch.cat([image, depth], dim=-3)
            
        else:
            image = transform(image)
            
        if actions is not None:  # training time
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                qpos, image, env_state
            )  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
