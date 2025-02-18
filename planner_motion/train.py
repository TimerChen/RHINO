import shutil
import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
import torch.optim as optim
import argparse
from collections import OrderedDict
from datasets import DataModule
from configs import get_config
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from models import *
from lightning.pytorch import loggers as pl_loggers

os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm

class LitProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.bar_format='{desc:<13}{percentage:3.0f}%|{bar:10}{r_bar}'
        return bar
    
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.bar_format='{desc:<13}{percentage:3.0f}%|{bar:10}{r_bar}'
        return bar
    
class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg
        self.mode = cfg.TRAIN.MODE
        
        self.his_length = cfg.TRAIN.HISTORY_LENGTH
        self.pred_length = cfg.TRAIN.PREDICT_LENGTH
        self.clip_length = cfg.TRAIN.HISTORY_LENGTH + cfg.TRAIN.PREDICT_LENGTH

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model

        self.writer = SummaryWriter(self.log_dir)

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()
    
    def _clip_motion(self, motion1, motion2, motion_lens):
        """
        Random clip a length of clip_length from the motion
            mask: his and pred are masked as 1.
            cond_maks : his is masked as 1.
        """
        hlen, plen = self.his_length, self.pred_length
        motion1 = motion1.detach().float()  # .to(self.device)
        motion2 = motion2.detach().float()  # .to(self.device)

        device = motion1.device
        B, T = motion1.shape[:2]
        
        # print("mask",motion_lens.device)
        mask = torch.arange(motion1.shape[1], device=device).reshape(1, -1).expand(B, -1) < motion_lens.reshape(-1, 1)
        mask = mask.float()

        # At least has one history frame and one pred frame
        mask_pad = torch.nn.functional.pad(mask.unsqueeze(-1), [0, 0, hlen-1, plen-1], mode='constant', value=0)[..., 0]
        motion1_pad = torch.nn.functional.pad(motion1, [0, 0, hlen-1, plen-1], mode='replicate')
        motion2_pad = torch.nn.functional.pad(motion2, [0, 0, hlen-1, plen-1], mode='replicate')
        i_s = torch.randint(0, motion1_pad.shape[1]+(hlen-1)-(hlen+plen), (B,), device=device)
        i_s = torch.remainder(i_s, motion_lens + (hlen-1) - (hlen + 1))
        # i_s[:] = motion_lens[0]+(hlen-1) - (hlen + 6)
        # print("mlen", motion_lens)
        # exit()
        indices = i_s.reshape(-1, 1) + torch.arange(hlen+plen, device=device).view(1, -1)
        mask = torch.gather(mask_pad, 1, indices)
        
        indices1 = indices.unsqueeze(-1).expand(-1, -1, motion1_pad.shape[-1])
        indices2 = indices.unsqueeze(-1).expand(-1, -1, motion2_pad.shape[-1])
        
        motion1 = torch.gather(motion1_pad, 1, indices1)
        motion2 = torch.gather(motion2_pad, 1, indices2)
        
        cond_mask = torch.zeros((B, hlen+plen), device=device).float()
        cond_mask[:, :self.his_length] = 1
        cond_mask = cond_mask * mask

        mask = torch.stack([mask.view(B, hlen+plen) ]*2, dim=-1)
        
         
        motions = torch.cat([motion1, motion2], dim=-1)
        # motion_lens = torch.minimum(motion_lens + hlen, i_s + plen) - i_s
        motion_lens = mask[..., 0].sum(dim=-1).long()
        # TOOD: check motion lens
        return motions, motion_lens, mask, cond_mask

    def _random_cond_mask(self, motion1, motion_lens):
        """ Generate a random mask for the history motion"""
        mini_pred = self.pred_length
        B = motion_lens.shape[0]

        # random gen a number in motion_lens, 1~motion1.shape[1]-mini_pred
        pos_shift = torch.randint(0, motion1.shape[1]-1, (B, ), device=motion1.device)

        # at least predict <mini_pred> frame
        max_start_pos = torch.maximum(motion_lens-mini_pred-1, torch.ones_like(motion_lens))
        # 1~motion_lens-mini_pred
        pos_shift = torch.remainder(pos_shift, max_start_pos).view(-1, 1) + 1
        cond_mask = torch.arange(motion1.shape[1], device=motion1.device).view(1, -1).expand(B, -1) < pos_shift.view(-1, 1)
        cond_mask = cond_mask.to(dtype=motion1.dtype)

        pos_shift_bit = torch.arange(0, motion1.shape[1], device=motion1.device).view(1, -1).expand(B, -1) + self.his_length - pos_shift
        
        # cond_mask (B, T), pos_shift (B, T)
        assert cond_mask.shape == motion1.shape[:2]
        assert pos_shift_bit.shape == motion1.shape[:2], f"{pos_shift_bit.shape} != {motion1.shape[:2]}"
        return cond_mask, pos_shift_bit

    def forward(self, batch_data):
        name, label, motion1, motion2, motion_lens, masks = batch_data
        
        mask, cond_mask = masks

        # idx = name.index("3292_swap")
        # print("check motion", idx, motion1[idx, 20, 16*3:18*3])
        
        print_check_tensor(motion1, "motion1")
        
        if torch.isnan(motion1).any() or torch.isnan(motion2).any():
            print("Nan in motion", name[torch.isnan(motion1).any(dim=1).any(dim=1)], name[torch.isnan(motion2).any(dim=1).any(dim=1)])
            motion1 = motion1.cpu()
            motion2 = motion2.cpu()
            idx = torch.arange(motion1.shape[0])[torch.isnan(motion1).any(dim=1).any(dim=1)]
            print("nan id", idx)
            print("nan text", [label[i] for i in idx])

            idx = torch.arange(motion2.shape[0])[torch.isnan(motion2).any(dim=1).any(dim=1)]
            print("nan id 2", idx)
            print("nan text 2", [label[i] for i in idx])
            assert False
            
        # print("motion shape", motion1.shape, motion2.shape, motion_lens.shape, masks[0].shape, masks[1].shape)
        
        # print("motion shape", motions.shape,mask.shape, cond_mask.shape)
        # exit()
        # motions = torch.cat([motion1, motion2], dim=-1)
        motions, motion_lens, mask, cond_mask = self._clip_motion(motion1, motion2, motion_lens)

        print_check_tensor(motions, "motion")
        if torch.isnan(motions).any():
            print("Nan in motion", name[torch.isnan(motions).any(dim=1).any(dim=1)])
            assert False

        # B = motion1.shape[0]
        # motions.reshape(B, self.his_length+self.pred_length, -1).type(torch.float32)
        # his_motions = motions[:, :self.his_length]
        # pred_motions = motions[:, self.his_length:]
        # motions = torch.cat([motion1, motion2], dim=-1)
        # cond_mask, pos_shift = self._random_cond_mask(motion1, motion_lens)

        batch = OrderedDict({})
        batch["label"] = label
        batch["motions"] = motions.float() # (B, T, 2*dim)
        batch["cond"] = motions.float() # (B, T, 2*dim)
        batch["cond_mask"] = cond_mask.float()
        batch["seq_mask"] = mask.float()
        batch["pos_shift"] = None # (B, T)
        batch["motion_lens"] = None

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()


    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()

        return {"loss": loss,
            "loss_logs": loss_logs}

    def validation_step(self, batch, batch_idx):
        loss, loss_info = self.forward(batch)
        self.log("val/loss", loss, )
        self.log("val/simple_loss", loss_info["simple"], )
        if loss_info.get("Human", None) is not None:
            self.log("val/Human_loss", loss_info["Human"], )
        self.log("val/H1_loss", loss_info["H1"], )
        print("validating...", loss_info)
        # self.validation_step_outputs.append(loss)
        return  {"loss": loss,
            "loss_logs": loss_info}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                self.writer.add_scalar(tag, mean_loss[tag], self.it)
                self.log(tag, mean_loss[tag])
            self.logs = OrderedDict()
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])



    def on_train_epoch_end(self):
        # pass
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()


    def save(self, file_name):
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return

class SaveConfCallback(Callback):
    def __init__(self, cfgs, save_dir):
        self.cfg = cfgs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        save_dir = self.save_dir
        print("++++++++++++ Log dir of config file", save_dir)
        with open(pjoin(save_dir, "model.yaml"), "w") as f:
            f.write(self.cfg[0].dump())
        with open(pjoin(save_dir, "train.yaml"), "w") as f:
            f.write(self.cfg[1].dump())
        with open(pjoin(save_dir, "data.yaml"), "w") as f:
            f.write(self.cfg[2].dump())
        
    

def build_models(cfg):
    if cfg.NAME == "InterGen":
        model = InterGen(cfg)
    return model


if __name__ == '__main__':
    print(os.getcwd())

    parse = argparse.ArgumentParser()
    parse.add_argument("--mconf", type=str, default="configs/model_train/d256x4_t300.yaml")
    parse.add_argument("--tconf", type=str, default="configs/train_30_10.yaml")
    parse.add_argument("--dtype", type=str, default="_6d")
    parse.add_argument("--backbone", type=str, default="_seq")
    parse.add_argument("--delay_shift", type=int, default=0)
    parse.add_argument("--no_diffusion", action="store_true")
    parse.add_argument("--no_human_motion", action="store_true")
    parse.add_argument("--scenario", type=int, default=2)
    parse.add_argument("--epiname", type=str, default="react")
    args = parse.parse_args()
    # args.delay_shift = 10
    name_suffix = f"{args.dtype}{args.backbone}_{args.delay_shift}_{args.epiname}_{args.scenario}"

    model_cfg = get_config(args.mconf[:-5] +args.dtype+ args.backbone +".yaml")
    train_cfg = get_config(args.tconf)
    # train_cfg = get_config("configs/train_15_15.yaml")

    data_cfg = get_config("configs/HH_datasets.yaml")["humanh1"+args.dtype]
    data_val_cfg = get_config("configs/HH_datasets.yaml")["humanh1_val"+args.dtype]
    data_cfg.defrost()
    data_cfg["DELAY_SHIFT"] = args.delay_shift
    data_val_cfg.defrost()
    data_val_cfg["DELAY_SHIFT"] = args.delay_shift
    if args.scenario == 2:
        data_cfg.DATA_ROOT = "./motion_data/motions_processed"
        data_cfg.SCENARIO = 2
        data_val_cfg.DATA_ROOT = "./motion_data/motions_processed"
        data_val_cfg.SCENARIO = 2
        model_cfg.defrost()
        model_cfg.COND_CLASS = 7
    if args.no_diffusion:
        model_cfg.defrost()
        model_cfg.DIFFUSION = False
    if args.no_human_motion:
        model_cfg.defrost()
        model_cfg.NO_HUMAN_MOTION = True
    
    # print("data cfg", data_cfg[args.dtype])
    print(f"Load model config from {args.mconf}, train config from {args.tconf}, data config from {args.dtype}")

    datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS, val_cfg=data_val_cfg, model_cfg=model_cfg)
    model = build_models(model_cfg)


    if train_cfg.TRAIN.RESUME:
        ckpt = torch.load(train_cfg.TRAIN.RESUME, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("checkpoint state loaded!")
    litmodel = LitTrainModel(model, train_cfg)

    print("Save model to path: ", litmodel.model_dir+name_suffix)
    os.makedirs(litmodel.model_dir+name_suffix, exist_ok=True)
    # shutil.copy("data_statistics_tmp.pkl", litmodel.model_dir+name_suffix+"/data_statistics.pkl")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir+name_suffix,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
                                                       save_top_k=train_cfg.TRAIN.SAVE_TOP_K,
                                                       save_last=True)
    conf_callback = SaveConfCallback([model_cfg, train_cfg, data_cfg], litmodel.model_dir+name_suffix)
    accelerator = accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    # accelerator = 'cpu'
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=litmodel.cfg.GENERAL.EXP_NAME, version=name_suffix)
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir+args.dtype,
        devices="auto", accelerator=accelerator,
        max_epochs=train_cfg.TRAIN.EPOCH,
        # strategy=DDPStrategy(find_unused_parameters=True),
        gpus=1,
        precision=32,
        check_val_every_n_epoch=50,
        callbacks=[checkpoint_callback, conf_callback, LitProgressBar(refresh_rate=1)],
        logger=tb_logger
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()
    
    with open(os.path.join(litmodel.model_dir+name_suffix, "data_statistics.pkl"), "wb") as f:
        print(f"[INFO] save data statistics to {os.path.join(litmodel.model_dir+name_suffix, 'data_statistics.pkl')}")
        pkl.dump(datamodule.train_dataset.data_statistics, f)
    
    model.set_normalizer({"stat": datamodule.train_dataset.data_statistics})
    trainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=valid_loader)
