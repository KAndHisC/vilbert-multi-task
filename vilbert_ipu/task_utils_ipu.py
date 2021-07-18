# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from io import open
import json
import logging
import numpy as np
from poptorch.optim import AdamW
from pytorch_transformers.optimization import WarmupConstantSchedule, WarmupLinearSchedule
import torch
import torch.distributed as dist
import poptorch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import DatasetMapTrain, DatasetMapEval
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import argparse
import json
from pytorch_transformers.optimization import (
    # AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)
from torch.optim.lr_scheduler import (
    LambdaLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
logger = logging.getLogger(__name__)


def LoadDatasets(args, task_cfg, task_ids, opts, split="trainval"):

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for _, task in enumerate(task_ids):
        if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
        if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )
    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_batch_size = {}
    task_num_iters = {}

    for _, task in enumerate(task_ids):
        task_name = task_cfg[task]["name"]
        batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps
        num_workers = args.num_workers
        # unsuported in IPU
        # if args.local_rank != -1:
        #     batch_size = int(batch_size / dist.get_world_size())
        #     num_workers = int(num_workers / dist.get_world_size())

        # num_workers = int(num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        task_datasets_train[task] = None
        if "train" in split:
            task_datasets_train[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
                split=task_cfg[task]["train_split"],
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ],
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
            )

        task_datasets_val[task] = None
        if "val" in split:
            task_datasets_val[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
                split=task_cfg[task]["val_split"],
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ],
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
            )

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        if "train" in split:
            # if args.local_rank == -1:
            # train_sampler = RandomSampler(task_datasets_train[task])
            # else:
            #     # TODO: check if this works with current data generator from disk that relies on next(file)
            #     # (it doesn't return item back by index)
            #     train_sampler = DistributedSampler(task_datasets_train[task])

            # task_dataloader_train[task] = DataLoader(
            #     task_datasets_train[task],
            #     sampler=train_sampler,
            #     batch_size=batch_size,
            #     num_workers=num_workers,
            #     pin_memory=True,
            # )
            ## TODO-- IPU
            # shuffle=train and not(isinstance(dataset, torch.utils.data.IterableDataset)),
            # drop_last=not(isinstance(dataset, torch.utils.data.IterableDataset)),
            # persistent_workers = True,
            # auto_distributed_partitioning = not isinstance(dataset, torch.utils.data.IterableDataset),
            # worker_init_fn=None,
            # mode=mode,
            # async_options={'load_indefinitely': True})
            task_dataloader_train[task] = poptorch.DataLoader(
                opts,
                task_datasets_train[task],
                # sampler=train_sampler,
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers,
                # pin_memory=True,
            )

            task_num_iters[task] = len(task_dataloader_train[task])
            task_batch_size[task] = batch_size

        if "val" in split:
            # task_dataloader_val[task] = DataLoader(
            #     task_datasets_val[task],
            #     shuffle=False,
            #     batch_size=batch_size,
            #     num_workers=2,
            #     pin_memory=True,
            # )
            task_dataloader_val[task] = poptorch.DataLoader(
                opts,
                task_datasets_val[task],
                shuffle=False,
                batch_size=batch_size,
                num_workers=2,
                # pin_memory=True,
            )

    return (
        task_batch_size,
        task_num_iters,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )

class FakeDataBuilder:
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
    def next(self):
        batch_size = self.batch_size
        return (
            torch.rand(batch_size, 4, 101, 2048), 
            torch.rand(batch_size, 4, 101, 5), 
            torch.ones([batch_size, 4, 101]).long(), 
            torch.randint(0, 10000, [batch_size, 4, 30]), 
            torch.zeros([batch_size]).long(), 
            torch.randint(0, 1, [batch_size, 4, 30]), 
            torch.zeros([batch_size, 4, 30]).long(), 
            torch.zeros([batch_size, 4, 101, 30]), 
            torch.randint(0, 10000, [batch_size])
        )

def GenFakeDatasets(args, task_cfg, task_ids, opts, split="trainval"):
    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_batch_size = {}
    task_num_iters = {}

    for _, task in enumerate(task_ids):
        batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps  
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        

        task_datasets_train[task] = None
        if "train" in split:
            task_dataloader_train[task] = FakeDataBuilder(batch_size)

        task_dataloader_val[task] = None
        if "val" in split:
            task_datasets_val[task] = FakeDataBuilder(batch_size)

        task_num_iters[task] = 100
        task_batch_size[task] = 0
        if "train" in split:
            task_batch_size[task] = batch_size
    return (
        task_batch_size,
        task_num_iters,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )


def LoadDatasetEval(args, task_cfg, task_ids, opts):

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True)

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task in enumerate(task_ids):
        if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
        if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_val = {}
    task_dataloader_val = {}
    task_batch_size = {}
    task_num_iters = {}

    for i, task in enumerate(task_ids):
        task_name = task_cfg[task]["name"]
        batch_size = args.batch_size
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())

        num_workers = int(args.num_workers / len(task_ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        if args.split:
            eval_split = args.split
        else:
            eval_split = task_cfg[task]["val_split"]

        task_datasets_val[task] = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=task_feature_reader1[
                task_cfg[task]["features_h5path1"]
            ],
            gt_image_features_reader=task_feature_reader2[
                task_cfg[task]["features_h5path2"]
            ],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            clean_datasets=args.clean_train_sets,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
        )

        # task_dataloader_val[task] = DataLoader(
        #     task_datasets_val[task],
        #     shuffle=False,
        #     batch_size=batch_size,
        #     num_workers=10,
        #     pin_memory=True,
        # )
        task_dataloader_val[task] = poptorch.DataLoader(
            opts,
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_val[task])
        task_batch_size[task] = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_datasets_val,
        task_dataloader_val,
    )


def GetParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, 
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir", default="save", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--config_file", default="config/bert_base_6layer_6conect.json", type=str,
                        help="The config file which specified the model details.")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--train_iter_multiplier", default=1.0, type=float, help="multiplier for the multi-task training.")
    parser.add_argument("--train_iter_gap", default=4, type=int,
                        help="forward every n iteration is the validation score is not improving over the last 3 epoch, -1 means will stop")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for." 
                        "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--do_lower_case", default=True, type=bool, 
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument( "--num_workers", type=int, default=8, help="Number of workers in the dataloader.")
    parser.add_argument( "--save_name", default="", type=str, help="save name for training.")
    parser.add_argument("--in_memory", default=False, type=bool, help="whether use chunck for parallel training.")
    parser.add_argument("--optim", default="AdamW", type=str, help="what to use for the optimization.")
    parser.add_argument("--tasks", default='8', type=str, help="1-2-3... training task separate by -")
    parser.add_argument("--freeze", default=-1, type=int, help="till which layer of textual stream of vilbert need to fixed.")
    parser.add_argument("--vision_scratch", action="store_true", help="whether pre-trained the image or not.")
    parser.add_argument("--evaluation_interval", default=1, type=int, help="evaluate very n epoch.")
    parser.add_argument("--lr_scheduler", default="mannul", type=str, help="whether use learning rate scheduler.")
    parser.add_argument("--baseline", action="store_true", help="whether use single stream baseline.")
    parser.add_argument("--resume_file", default="", type=str, help="Resume from checkpoint")
    parser.add_argument("--dynamic_attention", action="store_true", help="whether use dynamic attention.")
    parser.add_argument("--clean_train_sets", default=True, type=bool, help="whether clean train sets for multitask data.")
    parser.add_argument("--visual_target", default=0, type=int,
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.",
    )
    parser.add_argument("--task_specific_tokens", action="store_true", 
                        help="whether to use task specific tokens for the multi-task learning.")
    parser.add_argument("--enable_IPU",  action="store_true", help="whether use IPU to training.")
    parser.add_argument("--use_fake_data", action="store_true", help="whether use fake data to training.")
    # parser.add_argument(
    #     "--fp16",
    #     action="store_true",
    #     help="Whether to use 16-bit float precision instead of 32-bit",
    # )
    # parser.add_argument(
    #     "--loss_scale",
    #     type=float,
    #     default=0,
    #     help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
    #     "0 (default value): dynamic loss scaling.\n"
    #     "Positive power of 2: static loss scaling value.\n",
    # )
    return parser


def GetOptimizer(args, model, base_lr, median_num_iter):
    bert_weight_name = json.load(
        open("config/" + args.bert_model + "_weight_name.json", "r")
    )
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if "embeddings" in name:
                bert_weight_name_filtered.append(name)
            elif "encoder" in name:
                layer_num = name.split(".")[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)

        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False

        print("filtered weight")
        print(bert_weight_name_filtered)

    optimizer_grouped_parameters = []
    if len(list(model.named_parameters())) == 0:
        print('**** no model loaded! ****')
        exit()
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "vil_" in key:
                lr = 1e-4
            else:
                if args.vision_scratch:
                    if key[12:] in bert_weight_name:
                        lr = base_lr
                    else:
                        lr = 1e-4
                else:
                    lr = base_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]

    print(len(list(model.named_parameters())),
          len(optimizer_grouped_parameters))
    # if args.optim == "AdamW":
    optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr, bias_correction=False)
    # optimizer = AdamW(model.named_parameters(),  lr=base_lr, bias_correction=False)


    # elif args.optim == "RAdam":
    #     optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)
    num_train_optimization_steps = (
        median_num_iter * args.num_train_epochs // args.gradient_accumulation_steps
    )
    print("  Num steps: %d" % num_train_optimization_steps)
    warmpu_steps = args.warmup_proportion * num_train_optimization_steps

    if args.lr_scheduler == "warmup_linear":
        warmup_scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=warmpu_steps, t_total=num_train_optimization_steps
        )
    else:
        warmup_scheduler = WarmupConstantSchedule(
            optimizer, warmup_steps=warmpu_steps)

    lr_scheduler = None
    lr_reduce_list = np.array([5, 7])
    if args.lr_scheduler == "automatic":
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=1, cooldown=1, threshold=0.001
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=median_num_iter * args.num_train_epochs
        )
    elif args.lr_scheduler == "cosine_warm":
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=median_num_iter * args.num_train_epochs
        )
    elif args.lr_scheduler == "mannul":

        def lr_lambda_fun(epoch):
            return pow(0.2, np.sum(lr_reduce_list <= epoch))

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

    return optimizer, warmup_scheduler, lr_scheduler, lr_reduce_list, warmpu_steps,
