from typing import Any, List, Optional

import torch
from pytorch_lightning import LightningModule
from torchmetrics.functional.classification import (
    multilabel_accuracy,
    multilabel_f1_score,
)
from transformers import (
    AutoModel,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


class DietaryTagsClassifier(torch.nn.Module):
    """Dietary tags classifier class"""

    def __init__(
        self,
        model_config,
        num_tags: int,
        is_lm_model_pretrained: bool = True,
        freeze_lm: bool = True,
    ) -> None:
        """Classifier for dietary tags init method

        :param model_config: config for bert model
        :param num_tags: number of tags to be predicted
        :param is_lm_model_pretrained: whether , defaults to True
        """
        super().__init__()
        if is_lm_model_pretrained:
            self.model = AutoModel.from_pretrained(
                is_decoder=True, **model_config
            )
        else:
            self.model = AutoModel(model_config)
        if freeze_lm:
            for param in self.model.parameters():
                param.requires_grad = False
        self.classification_head = torch.nn.Linear(
            self.model.config.hidden_size, num_tags
        )

    def forward(self, model_input):
        """model forward pass

        :param model_input: input to bert model
        """
        out = self.model(**model_input).pooler_output
        out = self.classification_head(out)
        return out

    def predict(self, model_input):
        """predict dietary tags probabilities

        :param model_input: input to bert model
        """
        out = self.forward(model_input)
        return torch.sigmoid(out)


class DietaryTagsClassifierTrainer(LightningModule):
    """Trainer class for DietaryTagsClassifier"""

    def __init__(
        self,
        model_config: dict,
        tags_names: List[str],
        optimizer_params: dict,
        optimizer: str = "adam",
        learning_rate_scheduler_params: Optional[dict] = None,
        is_lm_model_pretrained: bool = True,
        freeze_lm: bool = True,
        pos_classes_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = DietaryTagsClassifier(
            model_config=model_config,
            num_tags=len(tags_names),
            is_lm_model_pretrained=is_lm_model_pretrained,
            freeze_lm=freeze_lm,
        )
        self.optimizer_params = optimizer_params
        self.learning_rate_scheduler_params = learning_rate_scheduler_params
        self.save_hyperparameters()
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_classes_weight)
        self.tags_names = tags_names
        self.optimizer_type = optimizer

    def configure_optimizers(self) -> Any:
        if self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW
        elif self.optimizer_type == "adam":
            optimizer = torch.optim.Adam
        else:
            raise Exception(f"Unknown optimizer type: '{optimizer}'")
        optimizer = optimizer(self.parameters(), **self.optimizer_params)
        if self.learning_rate_scheduler_params is not None:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=optimizer, **self.learning_rate_scheduler_params
            )
            return [optimizer,], [
                scheduler,
            ]
        return optimizer

    def _log_acc(
        self, y_pred, y_true, name, on_epoch=True, on_step=True, threshold=0.5
    ):

        acc = multilabel_accuracy(
            torch.sigmoid(y_pred),
            y_true.to(torch.int),
            num_labels=len(self.tags_names),
            threshold=threshold,
            average=None,
        )
        acc_avg = multilabel_accuracy(
            torch.sigmoid(y_pred),
            y_true.to(torch.int),
            num_labels=len(self.tags_names),
            threshold=threshold,
        )
        self.log(
            f"{name}_average_acc", acc_avg, on_epoch=on_epoch, on_step=on_step
        )
        for i, tag_name in enumerate(self.tags_names):
            self.log(
                f"{name}_{tag_name}_acc",
                acc[i],
                on_epoch=on_epoch,
                on_step=on_step,
            )

    def _log_f1(
        self, y_pred, y_true, name, on_epoch=True, on_step=True, threshold=0.5
    ):
        f1 = multilabel_f1_score(
            torch.sigmoid(y_pred),
            y_true.to(torch.int),
            num_labels=len(self.tags_names),
            threshold=threshold,
            average=None,
        )
        f1_avg = multilabel_f1_score(
            torch.sigmoid(y_pred),
            y_true.to(torch.int),
            num_labels=len(self.tags_names),
            threshold=threshold,
        )
        self.log(
            f"{name}_average_f1", f1_avg, on_epoch=on_epoch, on_step=on_step
        )
        for i, tag_name in enumerate(self.tags_names):
            self.log(
                f"{name}_{tag_name}_f1",
                f1[i],
                on_epoch=on_epoch,
                on_step=on_step,
            )

    def training_step(self, batch, *args):
        inputs, y_true = batch
        y_pred = self.forward(inputs)
        loss = self.loss(y_pred, y_true)
        self.log("training_loss", loss)
        self._log_acc(y_pred, y_true, "training")
        self._log_f1(y_pred, y_true, "training")
        return loss

    def validation_step(self, batch, *args: Any):
        with torch.inference_mode():
            inputs, y_true = batch
            y_pred = self.forward(inputs)
            loss = self.loss(y_pred, y_true)
            self.log("val_loss", loss)

        return y_pred, y_true

    def validation_epoch_end(self, outputs) -> None:
        y_pred, y_true = list(zip(*outputs))
        y_pred = torch.concat(y_pred, dim=0)
        y_true = torch.concat(y_true, dim=0)
        self._log_acc(y_pred, y_true, "val", on_step=False)
        self._log_f1(y_pred, y_true, "val", on_step=False)

    def forward(self, model_inputs, *args):
        return self.model(model_inputs)
