from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from transformers import BertModel
import torchmetrics


class DietaryTagsClassifier(torch.nn.Module):
    """Dietary tags classifier class"""

    def __init__(
        self,
        bert_config,
        num_tags: int,
        is_lm_model_pretrained: bool = True,
        freeze_lm: bool = True,
    ) -> None:
        """Classifier for dietary tags init method

        :param bert_config: config for bert model
        :param num_tags: number of tags to be predicted
        :param is_lm_model_pretrained: whether , defaults to True
        """
        super().__init__()
        if is_lm_model_pretrained:
            self.model = BertModel.from_pretrained(**bert_config)
        else:
            self.model = BertModel(bert_config)
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
        out = self.model(**model_input).last_hidden_state[:, 0]
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
        bert_config: dict,
        tags_names: List[str],
        optimizer_params: dict,
        is_lm_model_pretrained: bool = True,
        freeze_lm: bool = True,
    ):
        super().__init__()
        self.model = DietaryTagsClassifier(
            bert_config=bert_config,
            num_tags=len(tags_names),
            is_lm_model_pretrained=is_lm_model_pretrained,
            freeze_lm=freeze_lm,
        )
        self.optimizer_params = optimizer_params
        self.save_hyperparameters()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.tags_names = tags_names

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters(), **self.optimizer_params)

    def _log_acc(
        self, y_pred, y_true, name, on_epoch=True, on_step=True, threshold=0.5
    ):
        for i, tag_name in enumerate(self.tags_names):
            acc = torchmetrics.functional.accuracy(
                torch.sigmoid(y_pred[:, i]),
                y_true[:, i].to(torch.int),
                threshold=threshold,
            )
            self.log(
                f"{name}_{tag_name}_acc",
                acc,
                on_epoch=on_epoch,
                on_step=on_step,
            )

    def _log_f1(
        self, y_pred, y_true, name, on_epoch=True, on_step=True, threshold=0.5
    ):
        for i, tag_name in enumerate(self.tags_names):
            f1 = torchmetrics.functional.f1_score(
                torch.sigmoid(y_pred[:, i]),
                y_true[:, i].to(torch.int),
                threshold=threshold,
            )
            self.log(
                f"{name}_{tag_name}_f1", f1, on_epoch=on_epoch, on_step=on_step
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
            self._log_acc(y_pred, y_true, "val", on_step=False)
            self._log_f1(y_pred, y_true, "val", on_step=False)

        return loss

    def forward(self, model_inputs, *args):
        return self.model(model_inputs)
