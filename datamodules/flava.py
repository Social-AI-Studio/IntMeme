import torch

from torch.utils.data import DataLoader
from typing import Optional
from .utils import import_class

import lightning.pytorch as pl
from transformers import FlavaProcessor
from functools import partial

def flava_collate_fn(batch, processor, labels):    
    texts, images = [], []
    for item in batch:
        texts.append(item["text"])
        images.append(item["image"])
    
    inputs = processor(  
        text=texts, images=images, return_tensors="pt", padding=True, truncation=True
    )

    # Get Labels
    for l in labels:
        if l in batch[0].keys():
            labels = [feature[l] for feature in batch]
            inputs[l] = torch.tensor(labels, dtype=torch.int64)

    return inputs

class FlavaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cfg: str,
        processor_class_or_path: str,
        batch_size: int,
        shuffle_train: bool,
        num_workers: int
    ):
        super().__init__()

        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers= num_workers

        self.dataset_cls = import_class(dataset_cfg.dataset_class)

        processor = FlavaProcessor.from_pretrained(processor_class_or_path)
        self.collate_fn = partial(
            flava_collate_fn,
            processor=processor,
            labels=dataset_cfg.labels
        )
        
    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.train = self.dataset_cls(
                image_dir=self.dataset_cfg.image_dirs.train,
                annotation_filepath=self.dataset_cfg.annotation_filepaths.train,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.train,
                labels=self.dataset_cfg.labels,
                text_template=self.dataset_cfg.text_template
            )

            self.validate = self.dataset_cls(
                image_dir=self.dataset_cfg.image_dirs.validate,
                annotation_filepath=self.dataset_cfg.annotation_filepaths.validate,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.validate,
                labels=self.dataset_cfg.labels,
                text_template=self.dataset_cfg.text_template
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_cls(
                image_dir=self.dataset_cfg.image_dirs.test,
                annotation_filepath=self.dataset_cfg.annotation_filepaths.test,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.test,
                labels=self.dataset_cfg.labels,
                text_template=self.dataset_cfg.text_template
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_cls(
                image_dir=self.dataset_cfg.image_dirs.predict,
                annotation_filepath=self.dataset_cfg.annotation_filepaths.predict,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.predict,
                labels=self.dataset_cfg.labels,
                text_template=self.dataset_cfg.text_template
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)