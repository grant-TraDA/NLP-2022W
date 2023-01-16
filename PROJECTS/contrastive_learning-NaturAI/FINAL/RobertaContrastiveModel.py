# -*- coding: utf-8 -*-

# datasets==2.5.1
# numpy==1.23.3
# pandas==1.5.0
# pytorch_metric_learning==1.6.2
# scikit_learn==1.1.2
# sentence_transformers==2.2.2
# tensorflow==2.10.0
# torch==1.12.1
# transformers==4.22.2

import torch
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
import pandas as pd
from sklearn.utils import shuffle
from datasets import load_dataset
from transformers.file_utils import is_tf_available, is_torch_available
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
import transformers
from numpy import unique
from transformers import AutoTokenizer, Trainer, TrainingArguments, RobertaForSequenceClassification
from sklearn.model_selection import KFold
from pytorch_metric_learning import losses as losses_ml
from sentence_transformers.losses import BatchAllTripletLoss

"""# Utils - prepare dataset"""

def prepare_dataset(dataset="sst2", seed=42):
    if dataset == "sst2":
        df = pd.read_csv('/content/drive/MyDrive/NLP_Projekt/data/SST-2/train.tsv', sep='\t')
    elif dataset == "trec":
        dataset = load_dataset('trec')
        df = pd.DataFrame(
            list(zip([(eval['label-coarse']) for eval in dataset['train']],
                     [(eval['text']) for eval in dataset['train']])),
            columns=['label', 'sentence'])
    elif dataset == "mr":
        d = []
        with open('/content/drive/MyDrive/NLP_Projekt/data/MR/rt-polarity.neg', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('/content/drive/MyDrive/NLP_Projekt/data/MR/rt-polarity.pos', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
    elif dataset == "cr":
        d = []
        with open('/content/drive/MyDrive/NLP_Projekt/data/CR/custrev.neg', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('/content/drive/MyDrive/NLP_Projekt/data/CR/custrev.pos', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
    elif dataset == "mpqa":
        d = []
        with open('/content/drive/MyDrive/NLP_Projekt/data/MPQA/mpqa.neg', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('/content/drive/MyDrive/NLP_Projekt/data/MPQA/mpqa.pos', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
    elif dataset == "subj":
        d = []
        with open('/content/drive/MyDrive/NLP_Projekt/data/SUBJ/subj.objective', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('/content/drive/MyDrive/NLP_Projekt/data/SUBJ/subj.subjective', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
    elif dataset == "mrpc":
        df = pd.read_csv('/content/drive/MyDrive/NLP_Projekt/data/MRPC/train.tsv', sep='\t', error_bad_lines=False)
        df = df.rename(columns={'Quality': 'label', '#1 String': 'question', '#2 String': 'sentence'})
        df["question"] = df["question"].astype(str)
        df['sentence'] = df["sentence"].astype(str)
    elif dataset == 'imdb':
        df = pd.read_csv("/content/drive/MyDrive/NLP_Projekt/IMDB Dataset.csv").iloc[:2000,:]
        df = df.rename(columns={'review': 'sentence', 'sentiment': 'label'})
        df['label'][df.label == 'positive'] = 1
        df['label'][df.label == 'negative'] = 0
    else:
        raise ValueError(f'Cannot load the dataset: {dataset}.')
    df = shuffle(df, random_state=seed)
    return df

"""# Utils - training"""

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    return {
        'accuracy_score': acc,
        'f1_score': f1,
        'recall_score': recall,
        'precision_score': precision
    }

"""# Utils - triple entropy dataset"""

class CLDatasetClassification(torch.utils.data.Dataset):
    def __init__(self, index, df, tokenizer, max_length, sample_size=-1):
        if sample_size != -1:
            index = np.random.choice(index, sample_size, replace=False)

        texts = df.iloc[index]["sentence"].tolist()
        labels = df.iloc[index]["label"].tolist()
        train_encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

        self.encodings = train_encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class CLDatasetNLI(torch.utils.data.Dataset):
    def __init__(self, index, df, tokenizer, max_length, sample_size=-1):
        if sample_size != -1:
            index = np.random.choice(index, sample_size, replace=False)

        labels = df.iloc[index]["label"].tolist()
        questions = df.iloc[index]["question"].tolist()
        sentences = df.iloc[index]["sentence"].tolist()
        texts = list(zip(questions, sentences))
        texts = [txt for txt in texts if isinstance(txt[0], str) and isinstance(txt[1], str)]
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

"""# Roberta Contrastive"""

class RobertaContrastiveLearning(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()
        self.class_weight = kwargs.get('class_weights', None)
        self.clf_loss = kwargs.get('clf_loss', None)
        self.beta = kwargs.get('beta', None)
        self.only_cls = kwargs.get('only_cls', None)
        self.extended_inference = kwargs.get('extended_inference', None)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        if self.beta is not None:
            epsilon = self.beta
        else:
            epsilon = 1
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.config is not None:
                    loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                flatten_labels = torch.ones(sequence_output.shape[0] * sequence_output.shape[1]).type(
                    torch.LongTensor).to(loss.device)

                for label in labels.view(-1):
                    for idx, _ in enumerate(range(sequence_output.shape[1])):
                        flatten_labels[idx] = label

                sequence_selected = sequence_output.view(-1, sequence_output.shape[2])
                if self.clf_loss is not None:
                    if self.only_cls:
                        cl_loss = self.clf_loss(sequence_output[:, 0, :], labels.view(-1))
                    else:
                        cl_loss = self.clf_loss(sequence_selected, flatten_labels)

                    loss = epsilon * loss + (1 - epsilon) * cl_loss

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if self.extended_inference is not None and self.clf_loss is not None:
            sim_to_classes = self.clf_loss.get_logits(sequence_output[:, 0, :])
            softmax = torch.nn.Softmax(dim=1)
            logits = epsilon * softmax(logits) + (1 - epsilon) * softmax(sim_to_classes)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

"""# Cross validate"""

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def cross_validate(args):
    seed = args.seed
    max_length = args.max_length
    learning_rate = args.learning_rate
    num_warmup_steps = args.num_warmup_steps
    num_training_steps = args.num_training_steps
    eps = args.eps
    model_name = args.model_name
    model_type = args.model_type
    weight_decay = args.weight_decay
    la = args.la
    gamma = args.gamma
    margin = args.margin
    centers = args.centers
    beta = args.beta
    save_steps = args.save_steps
    sample_size = args.sample_size
    n_split = args.n_split
    supcon_temp = args.supcon_temp
    np.random.seed(seed)
    dataset_name = args.dataset_name
    softmax_scale = args.softmax_scale
    alpha = args.alpha
    extended_inference = args.extended_inference

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_steps=save_steps
    )

    kf = KFold(n_splits=n_split, random_state=seed, shuffle=True)
    cross_val_res = {}
    df = prepare_dataset(dataset_name)

    if model_name == "roberta-base":
        embedding_size = 768
    else:
        embedding_size = 1024

    for fold_id, (train_index, valid_index) in enumerate(kf.split(df)):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if dataset_name == "mrpc":
            train_dataset = CLDatasetNLI(train_index, df, tokenizer, max_length, sample_size)
            valid_dataset = CLDatasetNLI(valid_index, df, tokenizer, max_length)
        else:
            train_dataset = CLDatasetClassification(train_index, df, tokenizer, max_length, sample_size)
            valid_dataset = CLDatasetClassification(valid_index, df, tokenizer, max_length)
        if model_type == "softriple":
            clf_loss = losses_ml.SoftTripleLoss(num_classes=len(unique(df.label)), embedding_size=embedding_size,
                                             centers_per_class=centers, la=la, gamma=gamma, margin=margin)
        elif model_type == "supcon":
            clf_loss = losses_ml.SupConLoss(temperature=supcon_temp)
        elif model_type == "proxynca":
            clf_loss = losses_ml.ProxyNCALoss(len(unique(df.label)), embedding_size=embedding_size,
                                           softmax_scale=softmax_scale)
        elif model_type == "proxyanchor":
            clf_loss = losses_ml.ProxyAnchorLoss(len(unique(df.label)), embedding_size=embedding_size, margin=margin,
                                              alpha=alpha)
        elif model_type == "npairs":
            clf_loss = losses_ml.NPairsLoss()
        elif model_type == "triplet":
            clf_loss = BatchAllTripletLoss()
        elif model_type == "baseline":
            clf_loss = None
        else:
            raise ValueError(
                f'The model_type: {model_type} is not supported. Choose one of following: triple_entropy, supcon, baseline.')
        model = RobertaContrastiveLearning.from_pretrained(model_name,
                                                           num_labels=len(
                                                               unique(
                                                                   df.label)),
                                                           clf_loss=clf_loss,
                                                           beta=beta,
                                                           extended_inference=extended_inference)
        param_groups = [{"params": model.parameters(),
                         'lr': float(learning_rate)}]

        optimizer = transformers.AdamW(param_groups, eps=eps, weight_decay=weight_decay, correct_bias=True)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=num_warmup_steps,
                                                                 num_training_steps=num_training_steps)
        optimizers = optimizer, scheduler
        set_seed(seed)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            optimizers=optimizers,
            compute_metrics=compute_metrics
        )

        trainer.train()
        cross_val_res[fold_id] = trainer.evaluate()
        #trainer.save_model(args.output_dir + f'{fold_id}_{model_name}_{model_type}_{dataset_name}/model')
    results = []
    for measure in ["eval_f1_score", "eval_recall_score", "eval_accuracy_score", "eval_precision_score"]:
        m = np.mean([el[measure] for el in cross_val_res.values()])
        std = np.std([el[measure] for el in cross_val_res.values()])
        results.append([m,std])

    #print(f"Model type: {model_type}, Dataset name: {dataset_name}")
    #for measure in ["eval_f1_score", "eval_recall_score", "eval_accuracy_score", "eval_precision_score"]:
    #    print(
    #        f'measure: {measure.split("_")[1]}, mean: {np.mean([el[measure] for el in cross_val_res.values()])}, std: {np.std([el[measure] for el in cross_val_res.values()])}')
    return results, trainer