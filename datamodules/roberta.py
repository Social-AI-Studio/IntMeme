import torch

from typing import Optional
from .utils import import_class

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from functools import partial

def text_collate_fn(batch, tokenizer, labels):
    texts = []
    for item in batch:
        texts.append(item["text"])
    
    inputs = tokenizer(  
        text=texts, return_tensors="pt", padding=True
    )

    # Get Labels
    for l in labels:
        if l in batch[0].keys():
            labels = [feature[l] for feature in batch]

            if isinstance(labels[0], str):
                labels = tokenizer(labels, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False).input_ids
                labels[labels == tokenizer.pad_token_id] = -100
            else:
                labels = torch.tensor(labels, dtype=torch.int64)

            inputs[l] = labels

    return inputs

class RobertaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cfg: str,
        tokenizer_class_or_path: str,
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
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_class_or_path, use_fast=True)
        self.collate_fn = partial(text_collate_fn, tokenizer=tokenizer, labels=dataset_cfg.labels)
        
    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.train = self.dataset_cls(
                annotation_filepath=self.dataset_cfg.annotation_filepaths.train,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.train,
                labels=self.dataset_cfg.labels,
                input_template=self.dataset_cfg.text_template
            )

            self.validate = self.dataset_cls(
                annotation_filepath=self.dataset_cfg.annotation_filepaths.validate,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.validate,
                labels=self.dataset_cfg.labels,
                input_template=self.dataset_cfg.text_template
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_cls(
                annotation_filepath=self.dataset_cfg.annotation_filepaths.test,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.test,
                labels=self.dataset_cfg.labels,
                input_template=self.dataset_cfg.text_template
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_cls(
                annotation_filepath=self.dataset_cfg.annotation_filepaths.predict,
                auxiliary_dicts=self.dataset_cfg.auxiliary_dicts.predict,
                labels=self.dataset_cfg.labels,
                input_template=self.dataset_cfg.text_template
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)