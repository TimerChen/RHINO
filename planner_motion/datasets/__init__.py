import lightning.pytorch as pl
import torch
from .interhuman import HumanH1Dataset

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, batch_size, num_workers, val_cfg=None, test_cfg=None, model_cfg=None):
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.val_cfg = val_cfg
        self.test_cfg = test_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_cfg = model_cfg

    def setup(self, stage = None):
        """
        Create train and validation datasets
        """
        if self.cfg is not None:
            self.train_dataset = HumanH1Dataset(self.cfg, self.model_cfg)
        if self.val_cfg is not None:
            self.val_dataset = HumanH1Dataset(self.val_cfg, self.model_cfg)
        if self.test_cfg is not None:
            self.test_dataset = HumanH1Dataset(self.test_cfg, self.model_cfg)

    def train_dataloader(self):
        """
        Return train dataloader
        """
        # Calculate weights for each sample in the dataset
        class_counts = self.train_dataset.class_cnt
        
        def get_weight(data):
            if data["label"] == 0:
                cnt = class_counts[(data["label"], data["obj"][0], data["obj"][1])]
            else:
                cnt = class_counts[data["label"]]
            if cnt < 5:
                return 0.0
            return 1.0 / cnt

        weights = [get_weight(data) for data in self.train_dataset.data_list]
        extended_weights = weights * (len(self.train_dataset) // len(weights))
        sampler = torch.utils.data.WeightedRandomSampler(extended_weights, len(extended_weights))
        print("using weighted sampler")

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            sampler=sampler,
            drop_last=True,
            )
    
    def val_dataloader(self):
        """
        Return validation dataloader
        """
        print("val dataloader")
        if self.val_cfg is not None:
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=False,
                shuffle=False,
                drop_last=False,
                )
        else:
            return None
        
    def test_dataloader(self):
        """Return test dataloader"""
        if self.test_cfg is not None:
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=False,
                shuffle=False,
                drop_last=False,
                )
        else:
            return None
