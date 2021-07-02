# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from io import open
import logging
import torch.distributed as dist
import poptorch
from torch.utils.data import RandomSampler
# from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import DatasetMapTrain, DatasetMapEval
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader

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
            train_sampler = RandomSampler(task_datasets_train[task])
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
            task_dataloader_train[task] = poptorch.DataLoader(
                opts,
                task_datasets_train[task],
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
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
                pin_memory=True,
            )


    return (
        task_batch_size,
        task_num_iters,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )


def LoadDatasetEval(args, task_cfg, task_ids, opts):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

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





