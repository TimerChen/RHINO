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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, average_precision_score

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
    def __init__(self, model, cfg, scenario=0):
        super().__init__()
        # cfg init
        self.cfg = cfg
        self.mode = cfg.TRAIN.MODE
        self.scenario = scenario
        
        self.his_length = cfg.TRAIN.HISTORY_LENGTH
        self.pred_length = cfg.TRAIN.PREDICT_LENGTH
        self.clip_length = cfg.TRAIN.HISTORY_LENGTH + cfg.TRAIN.PREDICT_LENGTH

        self.hand_noise_std = cfg.TRAIN.HAND_NOISE_STD

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        self.validation_step_outputs = []

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model

        self.writer = SummaryWriter(self.log_dir)
        
        self.mismatch_matrix = {}
        self.predicted_label = {}

        self.all_true_labels = []
        self.all_predicted_labels = []

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()
    
    def _clip_motion(self, motion1, motion2, motion_lens, always_full_history=True, sample_start=0.0, test=False, hand_details=None):
        """
        Random clip a length of clip_length from the motion
            mask: his and pred are masked as 1.
            cond_maks : his is masked as 1.
        motion1 from (B, 200, 60) to (B, 15, 60)
        """
        hlen, plen = self.his_length, self.pred_length
        plen = 0
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
        
        # random gen start index
        if sample_start == 0:
            i_s = torch.randint(0, motion1_pad.shape[1]+(hlen-1)-(hlen+plen), (B,), device=device)
        else:
            p = sample_start  # 0.3
            random_choice = torch.rand((B,), device=device)
            i_s = torch.empty((B,), dtype=torch.long, device=device)
            # Sample from 0-15 with probability p
            i_s[random_choice < p] = torch.randint(hlen - 1, 16 + hlen - 1, ((random_choice < p).sum().item(), ), device=device)
            # Sample using the original method with probability 1-p
            i_s[random_choice >= p] = torch.randint(0, motion1_pad.shape[1] + (hlen - 1) - (hlen + plen), ((random_choice >= p).sum().item(),), device=device)
        
        if always_full_history:
            # padding length is hlen-1, so we right move the start index by hlen-1 to make sure the history is full
            assert (motion_lens + (hlen-1) - (hlen + 1) - (hlen-1) > 0).all()
            i_s = torch.remainder(i_s, motion_lens + (hlen-1) - (hlen + 1) - (hlen-1)) + (hlen-1)

        else:
            i_s = torch.remainder(i_s, motion_lens + (hlen-1) - (hlen + 1))
        # i_s[:] = motion_lens[0]+(hlen-1) - (hlen + 6)
        # print("mlen", motion_lens)
        # if test:
        #     i_s = torch.ones_like(i_s)
        #     i_s *= (8 + hlen - 1) # 200
        indices = i_s.reshape(-1, 1) + torch.arange(hlen+plen, device=device).view(1, -1)
        
        # if test:
        #     print("indices", indices-hlen+1)
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
        
        if hand_details is not None:
            hand, hand_img, hand_near = hand_details
            
            iarange = torch.arange(i_s.shape[0], device=device)
            hand = hand[iarange, i_s]
            hand_img = hand_img[iarange, i_s]
            hand_near = hand_near[iarange, i_s]
            hand_details = hand, hand_img, hand_near
            
            return motions, motion_lens, mask, cond_mask, hand_details
        
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

    def _add_hand_noise(self, motions):
        """
        motions: (B, T, 82=60+22)
        """
        motions[:, :, 48:60] += (self.hand_noise_std) * torch.randn_like(motions[:, :, 48:60])
        motions[:, :, -12:] += (self.hand_noise_std) * torch.randn_like(motions[:, :, -12:])
        return motions

    def forward(self, batch_data):
        
        ret = batch_data
        name, label, motion1, motion2, motion_lens, masks = ret[:6]
        mask, cond_mask = masks
        hand_details = ret[6:] if len(ret) > 6 else None
        
        print_check_tensor(motion1, "motion1")
        if torch.isnan(motion1).any() or torch.isnan(motion2).any():
            print("Nan in motion", name[torch.isnan(motion1).any(dim=1).any(dim=1)], name[torch.isnan(motion2).any(dim=1).any(dim=1)])
            motion1 = motion1.cpu()
            motion2 = motion2.cpu()
            idx = torch.arange(motion1.shape[0])[torch.isnan(motion1).any(dim=1).any(dim=1)]
            print("nan id", idx)

            idx = torch.arange(motion2.shape[0])[torch.isnan(motion2).any(dim=1).any(dim=1)]
            print("nan id 2", idx)
            assert False

        motions = torch.cat([motion1, motion2], dim=-1)
        # motions, motion_lens, mask, cond_mask, hand_details = self._clip_motion(motion1, motion2, motion_lens, hand_details=hand_details)

        print_check_tensor(motions, "motion")
        if torch.isnan(motions).any():
            print("Nan in motion", name[torch.isnan(motions).any(dim=1).any(dim=1)])
            assert False
        
        motions = self._add_hand_noise(motions)

        batch = OrderedDict({})
        batch["labels"] = label
        batch["motions"] = motions.float() # (B, T, 2*dim)
        batch["mask"] = mask.float()
        batch["motion_lens"] = None
        # hlist = ["hand_pos", "hand_imgs", "hand_near"]
        # for i, h in enumerate(hlist):
        #     if h in self.cfg.HD_INPUT:
        #         batch[h] = hand_details[i] if hand_details is not None else None
        batch["hand_pos"] = hand_details[0]
        batch["hand_imgs"] = hand_details[1]
        batch["hand_near"] = hand_details[2]
        batch["head_pos"] = hand_details[3]

        loss = self.model(batch)
        return loss

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.val_it = 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()


    def training_step(self, batch, batch_idx):
        loss, loss_info = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()
        loss_info.pop("label_pred")
        return {"loss": loss,
            "loss_logs": loss_info
            }

    def validation_step(self, batch, batch_idx):
        loss, loss_info = self.forward(batch)
        self.log("val_loss", loss, )
        self.log("val_accu", loss_info["accu"], )
        self.log("val_correct_no_idle", loss_info["correct_no_idle"])
        self.log("val_cnt_no_idle", loss_info["cnt_no_idle"])
        self.test_correct_no_idle += loss_info["correct_no_idle"]
        self.test_cnt_no_idle += loss_info["cnt_no_idle"]
        # print("label_pred", loss_info["label_pred"].shape)
        for v in loss_info["label_pred"]:
            self.mismatch_matrix[(v[0].item(), v[1].item())] = self.mismatch_matrix.get((v[0].item(), v[1].item()), 0) + 1
        loss_info.pop("label_pred")
        print("validating...", loss_info)
        # self.validation_step_outputs.append(loss)
        return  {"loss": loss,
            "loss_logs": loss_info
            }
        
    def test_step(self, batch, batch_idx):
        loss, loss_info = self.forward(batch)
        self.log("test_loss", loss, )
        self.log("test_accu", loss_info["accu"], )
        self.log("test_correct_no_idle", loss_info["correct_no_idle"])
        self.log("test_cnt_no_idle", loss_info["cnt_no_idle"])
        self.test_correct_no_idle += loss_info["correct_no_idle"]
        self.test_cnt_no_idle += loss_info["cnt_no_idle"]
        # print("label_pred", loss_info["label_pred"].shape)
        for i, v in enumerate(loss_info["label_pred"]):
            self.mismatch_matrix[(v[0].item(), v[1].item())] = self.mismatch_matrix.get((v[0].item(), v[1].item()), 0) + 1
            self.predicted_label[batch[0][i]] = v[1].item()
            self.all_true_labels.append(v[0].item())
            self.all_predicted_labels.append(v[1].item())
        loss_info.pop("label_pred")
        print("testing...", loss_info)
        return  {"loss": loss,
            "loss_logs": loss_info
            }
    
    # def on_validation_batch_end(self, outputs, batch, batch_idx, xxx):
         
    #     for k, v in outputs['loss_logs'].items():
    #         if k not in self.logs:
    #             self.logs[k] = v.item()
    #         else:
    #             self.logs[k] += v.item()
    #     # print("val loss", np.mean())
    #     print("callback on val end", self.it, self.cfg.TRAIN.LOG_STEPS)
    #     # do something with all preds
    #     # ...
    #     # self.validation_step_outputs.clear()  # free memory
    #     # if self.val_it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
    #     if self.val_it % self.cfg.TRAIN.LOG_STEPS == 0:
    #         mean_loss = OrderedDict({})
    #         for tag, value in self.logs.items():
    #             mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
    #             self.writer.add_scalar(tag, mean_loss[tag], self.it)
    #         self.logs = OrderedDict()
    #         print_current_loss(self.start_time, self.it, mean_loss,
    #                            self.trainer.current_epoch,
    #                            inner_iter=batch_idx,
    #                            lr=self.trainer.optimizers[0].param_groups[0]['lr'])



    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        # if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0:
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

    def on_test_epoch_start(self) -> None:
        self.mismatch_matrix = {}
        self.predicted_label = {}
        self.all_predicted_labels = []
        self.all_true_labels = []
        self.test_correct_no_idle = 0
        self.test_cnt_no_idle = 0
        
    def on_test_epoch_end(self) -> None:
        self.test_cnt_no_idle = max(self.test_cnt_no_idle, 1)
        self.log("test_accu_no_idle", self.test_correct_no_idle / self.test_cnt_no_idle)
        self._log_mismatch_matrix(self.trainer.current_epoch)
        self._log_predicted_label(self.trainer.current_epoch)
        self._calculate_mAP(self.trainer.current_epoch)
        
    def on_validation_epoch_start(self) -> None:
        self.mismatch_matrix = {}
        self.test_correct_no_idle = 0
        self.test_cnt_no_idle = 0
        
    def on_validation_epoch_end(self) -> None:
        self.test_cnt_no_idle = max(self.test_cnt_no_idle, 1)
        self.log("val_accu_no_idle", self.test_correct_no_idle / self.test_cnt_no_idle)
        if self.trainer.current_epoch % 5 == 0:
            self._log_mismatch_matrix(self.trainer.current_epoch)
            self._log_predicted_label(self.trainer.current_epoch)
    
    def _calculate_mAP(self, fid):
        if self.scenario == 0:
            cls_num = 13
        elif self.scenario == 1:
            cls_num = 12
        elif self.scenario == 2:
            cls_num = 17
        elif self.scenario == 3:
            cls_num = 15
        true = np.zeros((len(self.all_true_labels), cls_num))
        pred = np.zeros((len(self.all_predicted_labels), cls_num))
        for i, (true_label, pred_score) in enumerate(zip(self.all_true_labels, self.all_predicted_labels)):
            true[i, true_label] = 1
            pred[i, pred_score] = 1
        mAP = average_precision_score(true, pred, average='weighted')
        print("mAP: ", mAP)
        return mAP
        
    def _log_predicted_label(self, fid):
        with open(os.path.join(self.trainer.default_root_dir, f"predicted_label_{fid}.txt"), "w") as f:
            for k, v in self.predicted_label.items():
                f.write(f"{k} {v}\n")
        
    def _log_mismatch_matrix(self, fid):
        # write mismatch matrix as csv, and draw the confusion matrix
        if self.scenario == 0:
            cls_num = 13
            cls_name = {0: 'idle', 1: 'cheers', 2: 'thumbup', 3: 'handshake', 4: 'pick_can_R', 5: 'place_can_R', 
                        6: 'pick_tissue_L', 7: 'pick_table_plate_LR', 8: 'handover_plate_L', 9: 'get_human_plate_L',
                        10: 'wash_plate_LR', 11: 'place_plate_L', 12: 'place_sponge_R',-1:"full"}
        elif self.scenario == 1:
            cls_num = 12
            cls_name = {0: 'idle', 1: 'handshake', 2: 'thumbup', 3: 'get_cap_R', 4: 'give_cap_R', 5: 'pick_stamp_R', 
                        6: 'stamp_R', 7: 'place_stamp_R', 8: 'close_lamp', 9: 'open_lamp',
                        10: 'give_book_L', 11: 'cancel'}
        elif self.scenario == 2:
            cls_num = 17
            cls_name = {0: 'idle', 1: 'cheers', 2: 'thumbup', 3: 'handshake', 4: 'wave', 5: 'take_photo', 6: 'spread_hand', 
                        7: 'pick_can_R', 8: 'place_can_R', 9: 'pick_tissue_L', 10: 'pick_table_plate_LR', 11: 'handover_plate_L', 
                        12: 'get_human_plate_L', 13: 'wash_plate_LR', 14: 'place_plate_L', 15: 'place_sponge_R', 16: 'cancel', -1:"full"}
        elif self.scenario == 3:
            cls_num = 15
            cls_name = {0: 'idle', 1: 'thumbup', 2: 'handshake', 3: 'wave', 4: 'take_photo', 5: 'spread_hand', 6: 'get_cap_R', 
                        7: 'give_cap_R', 8: 'pick_stamp_R', 9: 'stamp_R', 10: 'place_stamp_R', 11: 'close_lamp', 
                        12: 'open_lamp', 13: 'give_book_L', 14: 'cancel', -1:"full"}
        with open(os.path.join(self.trainer.default_root_dir, f"mismatch_{fid}.csv"), "w") as f:
            for i in range(cls_num):
                for j in range(cls_num):
                    f.write(f"{self.mismatch_matrix.get((i, j), 0)},")
                f.write("\n")
        # draw confusion matrix
        y_true = []
        y_pred = []
        for k, v in self.mismatch_matrix.items():
            if k[0] == 0 and k[1] == 0:
                v = 0
            y_true = y_true + [k[0]]*v
            y_pred = y_pred + [k[1]]*v
        cm = confusion_matrix(y_true, y_pred,labels=list(range(cls_num)))
        if len(y_true) == 0:
            return
        plt.figure(figsize=(10, 7))
        plt.title("Confusion Matrix")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(np.arange(cls_num)+0.5, [cls_name[i] for i in range(cls_num)], rotation=45)
        plt.yticks(np.arange(cls_num)+0.5, [cls_name[i] for i in range(cls_num)], rotation=45)
        # enlarge bottom margin
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(os.path.join(self.trainer.default_root_dir, f"confusion_matrix_{fid}.png"))

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
        print("++++++++++++ Log dir of pl module", save_dir)
        with open(pjoin(save_dir, "model.yaml"), "w") as f:
            f.write(self.cfg[0].dump())
        with open(pjoin(save_dir, "train.yaml"), "w") as f:
            f.write(self.cfg[1].dump())
        with open(pjoin(save_dir, "data.yaml"), "w") as f:
            f.write(self.cfg[2].dump())
        
    

def build_model(cfg):
    if cfg.ADD_OCCUPANCY:
        input_dim = cfg.HUMAN_BODY_DIM + cfg.HAND_DIM + 2 * cfg.NUM_OBJ
    else:
        input_dim = cfg.HUMAN_BODY_DIM + cfg.HAND_DIM
    hand_input_dim = 0
    if "hand_pos" in cfg.HD_INPUT:
        hand_input_dim += 6
        if cfg.HAND_POS_ADD_Z:
            hand_input_dim += 2
    if "hand_near" in cfg.HD_INPUT:
        if cfg.BETTER_STATE:
            hand_input_dim += (cfg.NUM_OBJ+2)*2
        else:
            hand_input_dim += (cfg.NUM_OBJ+4)*2
    if "head_pos" in cfg.HD_INPUT:
        hand_input_dim += 1
    model = HumanClassifier(input_dim, cfg.NUM_CLASS, cfg.LATENT_DIM, 
                            ff_size=cfg.FF_SIZE, num_layers=cfg.NUM_LAYERS,
                            num_heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT, activation=cfg.ACTIVATION, 
                            cfg_weight=cfg.CFG_WEIGHT, skip_text=cfg.SKIP_TEXT,
                            hd_input=cfg.get("HD_INPUT", []), hand_input_dim=hand_input_dim)
    return model


if __name__ == '__main__':
    print(os.getcwd())

    parse = argparse.ArgumentParser()
    parse.add_argument("--mconf", type=str, default="configs/model_train/classifier.yaml")
    parse.add_argument("--tconf", type=str, default="configs/train_classifier.yaml")
    parse.add_argument("--dtype", type=str, default="react_1103_30fps_fullhis_handupdate6_class13")
    parse.add_argument("--hd_input", type=str, default=[], nargs="+")
    parse.add_argument("--no_train", action="store_true")
    parse.add_argument("--resume", type=str, default=None)
    parse.add_argument("--scenario", type=int, default=2)
    parse.add_argument("--obj-mask-ratio", type=float, default=None)
    parse.add_argument("--hand-noise-std", type=float, default=None)
    parse.add_argument("--better-state", action="store_true")
    parse.add_argument("--hand-iou-mean-pool", action="store_true")
    parse.add_argument("--add-z", action="store_true")
    parse.add_argument("--skip-occu", action="store_true")
    
    # parse.add_argument("--exp", type=str, default="3_and_idle")
    args = parse.parse_args()

    model_cfg = get_config(args.mconf)
    train_cfg = get_config(args.tconf)
    # train_cfg = get_config("configs/train_15_15.yaml")
    model_cfg.defrost()
    model_cfg.HAND_POS_ADD_Z = args.add_z
    model_cfg.HD_INPUT = args.hd_input
    model_cfg.BETTER_STATE = args.better_state
    model_cfg.HAND_IOU_MEAN_POOL = args.hand_iou_mean_pool
    if args.skip_occu:
        model_cfg.ADD_OCCUPANCY = False
    if args.obj_mask_ratio is not None:
        model_cfg.OBJ_MASK_RATIO = args.obj_mask_ratio
    if args.hand_noise_std is not None:
        model_cfg.HAND_NOISE_STD = args.hand_noise_std
    train_cfg.defrost()
    train_cfg.TRAIN.HISTORY_LENGTH = model_cfg.HISTORY_LENGTH
    train_cfg.TRAIN.PREDICT_LENGTH = model_cfg.PREDICT_LENGTH
    assert model_cfg.PREDICT_LENGTH == 0, f"predict length should be 0, but got {model_cfg.PREDICT_LENGTH}"

    if not args.no_train:
        data_cfg = get_config("configs/HH_datasets.yaml")["interhuman"]
        if args.scenario == 1:
            data_cfg.defrost()
            data_cfg.DATA_ROOT = "./mydata2/motions_processed"
            data_cfg.SCENARIO = 1
        elif args.scenario == 2:
            data_cfg.defrost()
            data_cfg.DATA_ROOT = "../motion_data/motions_processed"
            data_cfg.SCENARIO = 2
        elif args.scenario == 3:
            data_cfg.defrost()
            data_cfg.DATA_ROOT = "./mydata4/motions_processed"
            data_cfg.SCENARIO = 3
    else:
        data_cfg = None
    data_val_cfg = get_config("configs/HH_datasets.yaml")["interhuman_val"]
    data_test_cfg = get_config("configs/HH_datasets.yaml")["interhuman_test"]
    if args.scenario == 1:
        data_test_cfg.defrost()
        data_test_cfg.DATA_ROOT = "./mydata2/motions_processed"
        data_test_cfg.SCENARIO = 1
        model_cfg.NUM_CLASS = 12
    elif args.scenario == 2:
        data_test_cfg.defrost()
        data_test_cfg.DATA_ROOT = "./motion_data/motions_processed"
        data_test_cfg.SCENARIO = 2
        model_cfg.NUM_CLASS = 17
    elif args.scenario == 3:
        data_test_cfg.defrost()
        data_test_cfg.DATA_ROOT = "./mydata4/motions_processed"
        data_test_cfg.SCENARIO = 3
        model_cfg.NUM_CLASS = 15
    # print("data cfg", data_cfg[args.dtype])
    print(f"Load model config from {args.mconf}, train config from {args.tconf}, data config from {args.dtype}")

    if not args.no_train:
        datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS, val_cfg=data_test_cfg, model_cfg=model_cfg, test_cfg=data_test_cfg)
    else:
        datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS, val_cfg=data_test_cfg, model_cfg=model_cfg, test_cfg=data_test_cfg)
    # datamodule = DataModule(data_test_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS, val_cfg=data_test_cfg, model_cfg=model_cfg)
    model = build_model(model_cfg)

    if train_cfg.TRAIN.RESUME:
        ckpt = torch.load(train_cfg.TRAIN.RESUME, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("checkpoint state loaded!")
    litmodel = LitTrainModel(model, train_cfg, args.scenario)

    print("Save model to path: ", litmodel.model_dir+args.dtype)
    os.makedirs(litmodel.model_dir+args.dtype, exist_ok=True)


    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir+args.dtype,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
                                                       save_top_k=train_cfg.TRAIN.SAVE_TOP_K,
                                                       save_last='link')
    conf_callback = SaveConfCallback([model_cfg, train_cfg, data_cfg], litmodel.model_dir+args.dtype)
    accelerator = accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    # accelerator = "cpu"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=litmodel.cfg.GENERAL.EXP_NAME, version=args.dtype)
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir+args.dtype,
        devices="auto", accelerator=accelerator,
        max_epochs=train_cfg.TRAIN.EPOCH,
        # strategy=DDPStrategy(find_unused_parameters=True),
        gpus=1,
        precision=32,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, conf_callback, LitProgressBar(refresh_rate=1)],
        logger=tb_logger,
    )
    datamodule.setup()
    valid_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    if not args.no_train:
        train_loader = datamodule.train_dataloader()
        # shutil.copy("data_statistics_tmp.pkl", litmodel.model_dir+args.dtype+"/data_statistics.pkl")
        with open(os.path.join(litmodel.model_dir+args.dtype, "data_statistics.pkl"), "wb") as f:
            pkl.dump(datamodule.train_dataset.data_statistics, f)
        model.set_normalizer({"stat": datamodule.train_dataset.data_statistics})
    else:
        train_loader = None
        with open(os.path.join(litmodel.model_dir+args.dtype,"data_statistics.pkl"), "rb") as f:
            data_statistics = pkl.load(f)
        model.set_normalizer({"stat": data_statistics})

    if args.no_train:
        ckpt_path = os.path.join(litmodel.model_dir+args.dtype, args.resume)
        # trainer.validate(model=litmodel, dataloaders=valid_loader, ckpt_path=ckpt_path)
        trainer.test(model=litmodel, dataloaders=test_loader, ckpt_path=ckpt_path)
        
    else:
        trainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=valid_loader)
