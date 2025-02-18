import copy
import os.path
import sys
import torch
import lightning as L
import scipy.ndimage.filters as filters

from os.path import join as pjoin
from models import *
from collections import OrderedDict
from configs import get_config
from utils.plot_script import *
from utils.preprocess import *
from utils import paramUtil
from datasets import DataModule

class LitGenModel(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')
        self.result_path = self.cfg.GENERAL.RESULT_PATH

        self.his_length = cfg.TRAIN.HISTORY_LENGTH
        self.pred_length = cfg.TRAIN.PREDICT_LENGTH
        self.clip_length = cfg.TRAIN.HISTORY_LENGTH + cfg.TRAIN.PREDICT_LENGTH

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # train model init
        self.model = model

        # others init
        self.normalizer = MotionNormalizer()
        self.normalizer_torch = MotionNormalizerTorch()

    def plot_motion_pose(self, motions, result_path, caption):
        """  motions: (T, 2, D) """
        mp_data = self.extract_pose(motions)

        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))

        mp_joint = []
        for i, data in enumerate(mp_data):
            if i == 0:
                joint = data[:,:22*3].reshape(-1,22,3)
            else:
                joint = data[:,:22*3].reshape(-1,22,3)

            mp_joint.append(joint)

        print("save to: ", result_path)
        with open("plot_joint.pkl", "wb") as f:
            import pickle
            pickle.dump(mp_joint, f)

        plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)


    def generate_one_sample(self, prompt, window_size, prompt_mask, ref_motions):
        self.model.eval()
        batch = OrderedDict({})

        batch["motion_lens"] = torch.zeros(1,1).long().cuda()
        batch["prompt"] = prompt
        batch["cond_mask"] = prompt_mask

        # window_size = 210
        motion_output_raw = self.generate_loop(batch, window_size, ref_motions)
        
        # (T, 2, D)
        return motion_output_raw

    def generate_loop(self, batch, window_size, ref_motions):
        prompt = batch["prompt"]
        batch = copy.deepcopy(batch)
        batch["motion_lens"][:] = window_size

        if self.model.skip_text:
            prompt = self.normalizer_torch.forward(prompt.reshape( prompt.shape[:2]+(2, -1)))
            # cond = kwargs["model_kwargs"]["cond"].reshape(target.shape)
            # kwargs["model_kwargs"]["cond"] = self.normalizer.forward(cond).reshape(target.shape[:2] + (-1,))
            batch["cond"] = prompt
        else:
            batch["text"] = [prompt]
        
        batch = self.model.forward_test(batch)
        motion_output_both = batch["output"][0].reshape(batch["output"][0].shape[0], 2, -1)
        motion_output_both = self.normalizer.backward(motion_output_both.cpu().detach().numpy())
        # motion_output_both = motion_output_both.cpu().detach().numpy()

        # convert upper motion to full body motion

        m0 = recover_upper2full(motion_output_both[:,0], ref_motions[0]).reshape(batch["output"][0].shape[0], 1, -1)
        m1 = recover_upper2full(motion_output_both[:,1], ref_motions[1]).reshape(batch["output"][0].shape[0], 1, -1)

        motion_output_both = np.concatenate([m0, m1], axis=1)

        # only use pred history
        motion_output_both = motion_output_both[self.his_length:]

        return motion_output_both

    def extract_pose(self, motions):
        """  motions [(T, D)]*2  """
        sequences = [None, None]

        for j in range(2):
            motion_output = motions[j]
            
            # motion_output = self.normalizer.backward(motion_output, full=True)

            joints3d = motion_output[:,:22*3].reshape(-1,22,3)
            # joints3d = filters.gaussian_filter1d(joints3d, 1, axis=0, mode='nearest')
            # sequences[j].append(joints3d)
            sequences[j] = joints3d

        # sequences[0] = np.concatenate(sequences[0], axis=0)
        # sequences[1] = np.concatenate(sequences[1], axis=0)
        return sequences

def build_models(cfg):
    if cfg.NAME == "InterGen":
        model = InterGen(cfg)
    return model

hand_chains = [
    # kinematic_chain = [[0, 2, 5, 8, 11],
    #              [0, 1, 4, 7, 10],
    #              [0, 3, 6, 9, 12, 15],
    [17, 19, 21],
    [16, 18, 20]
]
hand_joints = [19, 21] + [18, 20]

def rule_rectify(cond, motions, ref_motions):
    """ get vel by rule 
        cond: (T, 22*3)
        motions: (T, 22*3)

    """
    # assert cond.shape == ref_motions.shape, f"shape not match: {cond.shape} != {ref_motions.shape}"
    motions_pos = motions[:, :22*3].reshape(-1, 22, 3)
    motions_vel = motions_pos[1:] - motions_pos[:-1]

    # [New] fix joints that are not in hand
    nohand_idx = [i for i in range(22) if i not in hand_joints]
    for i in nohand_idx:
        motions_vel[:, i:(i+1)] = 0

    # use vel to recover pos
    motions_pos[0] = (cond[-1, :22*3] + cond[-1, 22*3:22*6]).reshape(22, 3)
    for i in range(1, motions_pos.shape[0]):
        motions_pos[i] = motions_pos[i-1] + motions_vel[i-1]

    # motions[:, :22*3] = motions_pos.reshape(-1, 22*3)
    motions[:-1, 22*3:22*6] = motions_vel.reshape(-1, 22*3)

    # [New] calc body lengh on hand
    mpos = motions[:, :22*3].reshape(-1, 22, 3)
    motion2 = mpos.copy()
    # from models.losses import kinematic_chain
    ref_motions = ref_motions[:, :22*3].reshape(1, 22, 3)
    
    for i, chain in enumerate(hand_chains):
        # motion[:, chain[1:]] -= motion[:, chain[0]].reshape(-1, 1, 3)
        # print("ref_motions", ref_motions.shape, motions.shape)
        for j in range(1, len(chain)):
            dir_vec = (mpos[:, chain[j]] - mpos[:, chain[j-1]]) / np.linalg.norm(mpos[:, chain[j]] - mpos[:, chain[j-1]], axis=-1, keepdims=True)
            vec_len = np.linalg.norm(ref_motions[:, chain[j]] - ref_motions[:, chain[j-1]], axis=-1, keepdims=True)
            motion2[:, chain[j]] = motion2[:, chain[j-1]] + dir_vec * vec_len
            aaa = np.linalg.norm(mpos[:, chain[j]] - mpos[:, chain[j-1]], axis=-1, keepdims=True)
        #     print(f" {chain[j]}->{chain[j-1]}: {aaa.shape}", end="")
        #     print(f" {chain[j]}->{chain[j-1]}: {vec_len.shape}", end="")
        # print("")
    motions[:, :22*3] = motion2.reshape(-1, 22*3)
        
    # [New] re-calc hand vel
    motions_pos = motions[:, :22*3].reshape(-1, 22, 3)
    motions_vel = motions_pos[1:] - motions_pos[:-1]
    motions[:-1, 22*3:22*6] = motions_vel.reshape(-1, 22*3)

    return motions