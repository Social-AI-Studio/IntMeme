import torch
import importlib
import logging
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from transformers import AutoModel
# from .model_utils import setup_metrics
import torchmetrics


def setup_metrics(obj, cls_dict, metrics_cfg, stage):
    for cls_name, num_classes in cls_dict.items():
        for cfg in metrics_cfg.values():
            metric_name = cfg["name"]
            metric_class = getattr(torchmetrics, metric_name)

            # e.g., train_validate_auroc
            setattr(
                obj,
                f"{stage}_{cls_name}_{metric_name.lower()}",
                metric_class(num_classes=num_classes, task=cfg.task)
            )


class MimeModel(pl.LightningModule):
    def __init__(
        self,
        multimodal_class_or_path: str,
        rc_class_or_path: str,
        metrics_cfg: dict,
        cls_dict: dict,
        optimizers: list
    ):
        super().__init__()
        self.save_hyperparameters()

        self.multimodal_model = AutoModel.from_pretrained(multimodal_class_or_path)
        self.rc_model = AutoModel.from_pretrained(rc_class_or_path)

        self.metric_names = [cfg.name.lower() for cfg in metrics_cfg.values()]
        self.cls_dict = cls_dict
        self.optimizers = optimizers

        # set up classification
        combined_hidden_size = self.multimodal_model.config.multimodal_config.hidden_size + self.rc_model.config.hidden_size
        self.mlps = nn.ModuleList([
            nn.Linear(combined_hidden_size, num_classes)
            for num_classes in cls_dict.values()
        ])

        # set up metric
        setup_metrics(self, cls_dict, metrics_cfg, "train")
        setup_metrics(self, cls_dict, metrics_cfg, "validate")
        setup_metrics(self, cls_dict, metrics_cfg, "test")

    def compute_metrics_step(self, cls_name, stage, loss, targets, preds):
        self.log(f'{stage}_{cls_name}_loss', loss, prog_bar=True)

        for metric_name in self.metric_names:
            metric = getattr(self, f"{stage}_{cls_name}_{metric_name}")
            metric(preds, targets)

    def compute_metrics_epoch(self, cls_name, stage):
        msg = "Epoch Results:\n"

        avg_metric_score = 0
        for metric_name in self.metric_names:
            metric = getattr(self, f"{stage}_{cls_name}_{metric_name}")
            metric_score = metric.compute()
            avg_metric_score += metric_score

            self.log(f'{stage}_{cls_name}_{metric_name}', metric_score,
                     prog_bar=True, sync_dist=True)
            
            msg += f"\t{stage}_{cls_name}_{metric_name}: {metric_score}\n"

            # reset the metrics
            metric.reset()


        avg_metric_score = avg_metric_score / len(self.metric_names)
        
        self.log(f'{stage}_{cls_name}_average', avg_metric_score,
                    prog_bar=True, sync_dist=True)
        
        msg += f"\t{stage}_{cls_name}_average: {avg_metric_score}\n"

        logging.info(msg)


    def training_step(self, batch, batch_idx):
        multimodel_outputs = self.multimodal_model(
            input_ids=batch["meme_input_ids"],
            attention_mask=batch["meme_attention_mask"],
            pixel_values=batch['pixel_values']
        )

        rc_outputs = self.rc_model(
            input_ids=batch["passage_input_ids"],
            attention_mask=batch["passage_attention_mask"]
        )

        total_loss = 0.0
        
        for idx, cls_name in enumerate(self.cls_dict.keys()):
            targets = batch[cls_name]

            combined_hidden_state = torch.cat(
                (multimodel_outputs.multimodal_embeddings[:, 0], rc_outputs.last_hidden_state[:, 0]),
                1
            )
            preds = self.mlps[idx](combined_hidden_state)

            loss = F.cross_entropy(preds, targets)
            total_loss += loss
            
            self.compute_metrics_step(
                cls_name, "train", loss, targets, preds)


        return total_loss / len(self.cls_dict)

    def on_training_epoch_end(self):
        for cls_name in self.cls_dict.keys():
            self.compute_metrics_epoch(cls_name, "train")

    def validation_step(self, batch, batch_idx):
        multimodel_outputs = self.multimodal_model(
            input_ids=batch["meme_input_ids"],
            attention_mask=batch["meme_attention_mask"],
            pixel_values=batch['pixel_values']
        )

        rc_outputs = self.rc_model(
            input_ids=batch["passage_input_ids"],
            attention_mask=batch["passage_attention_mask"]
        )

        total_loss = 0.0
        
        for idx, cls_name in enumerate(self.cls_dict.keys()):
            targets = batch[cls_name]

            combined_hidden_state = torch.cat(
                (multimodel_outputs.multimodal_embeddings[:, 0], rc_outputs.last_hidden_state[:, 0]),
                1
            )
            preds = self.mlps[idx](combined_hidden_state)

            loss = F.cross_entropy(preds, targets)
            total_loss += loss

            self.compute_metrics_step(
                cls_name, "validate", loss, targets, preds)

            total_loss += loss

        return total_loss / len(self.cls_dict)

    def on_validation_epoch_end(self):
        for cls_name in self.cls_dict.keys():
            self.compute_metrics_epoch(cls_name, "validate")

    def test_step(self, batch, batch_idx):
        multimodel_outputs = self.multimodal_model(
            input_ids=batch["meme_input_ids"],
            attention_mask=batch["meme_attention_mask"],
            pixel_values=batch['pixel_values']
        )

        rc_outputs = self.rc_model(
            input_ids=batch["passage_input_ids"],
            attention_mask=batch["passage_attention_mask"]
        )

        total_loss = 0.0
        
        for idx, cls_name in enumerate(self.cls_dict.keys()):
            targets = batch[cls_name]

            combined_hidden_state = torch.cat(
                (multimodel_outputs.multimodal_embeddings[:, 0], rc_outputs.last_hidden_state[:, 0]),
                1
            )
            preds = self.mlps[idx](combined_hidden_state)

            loss = F.cross_entropy(preds, targets)
            total_loss += loss

            self.compute_metrics_step(
                cls_name, "test", loss, targets, preds)

            total_loss += loss

        return total_loss / len(self.cls_dict)

    def on_test_epoch_end(self):
        for cls_name in self.cls_dict.keys():
            self.compute_metrics_epoch(cls_name, "test")

    def predict_step(self, batch, batch_idx):
        multimodel_outputs = self.multimodal_model(
            input_ids=batch["meme_input_ids"],
            attention_mask=batch["meme_attention_mask"],
            pixel_values=batch['pixel_values']
        )

        rc_outputs = self.rc_model(
            input_ids=batch["passage_input_ids"],
            attention_mask=batch["passage_attention_mask"]
        )

        results = {}
        for idx, cls_name in enumerate(self.cls_dict.keys()):
            # targets = batch[cls_name]

            combined_hidden_state = torch.cat(
                (multimodel_outputs.multimodal_embeddings[:, 0], rc_outputs.last_hidden_state[:, 0]),
                1
            )
            preds = self.mlps[idx](combined_hidden_state)
            
            # results["img"] = batch["image_filename"].tolist()
            # results[f"{cls_name}_logits"] = preds.tolist()
            # results[f"{cls_name}_preds"] = torch.argmax(preds, dim=1).tolist()
            # results[f"{cls_name}_labels"] = batch[cls_name].tolist()
            # results['hate'] = preds

        return preds

    def configure_optimizers(self):
        opts = []
        for opt_cfg in self.optimizers:
            class_name = opt_cfg.pop("class_path")
            
            package_name = ".".join(class_name.split(".")[:-1])
            package = importlib.import_module(package_name)
            
            class_name = class_name.split(".")[-1]
            cls = getattr(package, class_name)

            opts.append(cls(self.parameters(), **opt_cfg))

        return opts
