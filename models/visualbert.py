import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import torchmetrics
import importlib
from transformers import VisualBertModel

from datamodules.collators.gqa_lxmert.modeling_frcnn import GeneralizedRCNN
from datamodules.collators.gqa_lxmert.lxmert_utils import Config

from .model_utils import setup_metrics
from .mixins import MetricMixin


class VisualBertClassificationModel(pl.LightningModule, MetricMixin):
    def __init__(
        self,
        model_class_or_path,
        metrics_cfg: dict,
        cls_dict,
        frcnn_class_or_path=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VisualBertModel.from_pretrained(model_class_or_path)
        if frcnn_class_or_path:
            self.frcnn_cfg = Config.from_pretrained(
                "unc-nlp/frcnn-vg-finetuned")
            self.frcnn = GeneralizedRCNN.from_pretrained(
                "unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)

        # set up classification
        self.mlps = nn.ModuleList([
            nn.Linear(self.model.config.hidden_size, value)
            for value in cls_dict.values()
        ])

        # set up metric
        self.cls_dict = cls_dict

        # set up metric
        setup_metrics(self, cls_dict, metrics_cfg, "train")
        setup_metrics(self, cls_dict, metrics_cfg, "validate")
        setup_metrics(self, cls_dict, metrics_cfg, "test")

    def training_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        if "visual_feats" in batch and "visual_pos" in batch:
            visual_feats = batch['visual_feats']
        else:
            # Run Faster-RCNN
            images = batch['images']
            sizes = batch['sizes']
            scales_yx = batch['scales_yx']

            visual_dict = self.frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt",
            )

            visual_feats = visual_dict['visual_feats']

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_feats,
            visual_attention_mask=torch.ones(
                visual_feats.shape[:-1]).to('cuda'),
            token_type_ids=token_type_ids,
        )

        total_loss = 0

        for idx, cls_name in enumerate(self.cls_dict.keys()):
            targets = batch[cls_name]
            preds = self.mlps[idx](model_outputs.last_hidden_state[:, 0, :])

            loss = F.cross_entropy(preds, targets)
            total_loss += loss

            self.compute_metrics_step(
                cls_name, "train", loss, targets, preds)

        return total_loss / len(self.cls_dict)

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        if "visual_feats" in batch and "visual_pos" in batch:
            visual_feats = batch['visual_feats']
        else:
            # Run Faster-RCNN
            images = batch['images']
            sizes = batch['sizes']
            scales_yx = batch['scales_yx']

            visual_dict = self.frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt",
            )

            visual_feats = visual_dict['visual_feats']

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_feats,
            visual_attention_mask=torch.ones(
                visual_feats.shape[:-1]).to('cuda'),
            token_type_ids=token_type_ids,
        )

        total_loss = 0

        for idx, cls_name in enumerate(self.cls_dict.keys()):
            targets = batch[cls_name]
            preds = self.mlps[idx](model_outputs.last_hidden_state[:, 0, :])

            loss = F.cross_entropy(preds, targets)
            total_loss += loss

            self.compute_metrics_step(
                cls_name, "validate", loss, targets, preds)

        return total_loss / len(self.cls_dict)

    def test_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        visual_feats = batch['visual_feats']
        token_type_ids = batch['token_type_ids']

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_feats,
            visual_attention_mask=torch.ones(
                visual_feats.shape[:-1]).to('cuda'),
            token_type_ids=token_type_ids,
        )

        total_loss = 0

        for idx, cls_name in enumerate(self.cls_dict.keys()):
            targets = batch[cls_name]
            preds = self.mlps[idx](model_outputs.last_hidden_state[:, 0, :])

            loss = F.cross_entropy(preds, targets)
            total_loss += loss

            self.compute_metrics_step(
                cls_name, "test", loss, targets, preds)

        return total_loss / len(self.cls_dict)

    def predict_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        visual_feats = batch['visual_feats']
        token_type_ids = batch['token_type_ids']

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_feats,
            visual_attention_mask=torch.ones(visual_feats.shape[:-1]),
            token_type_ids=token_type_ids,
        )

        results = {}
        for idx, cls_name in enumerate(self.cls_dict.keys()):
            preds = self.mlps[idx](model_outputs.last_hidden_state[:, 0, :])

            results["img"] = batch["image_filename"].tolist()
            results[f"{cls_name}_preds"] = torch.argmax(preds, dim=1).tolist()
            results[f"{cls_name}_labels"] = batch[cls_name].tolist()

        return results

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]
