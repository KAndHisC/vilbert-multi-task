# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import random
from io import open
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect


import yaml
from easydict import EasyDict as edict

import pdb
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

import poptorch
        


from poptorch.optim import AdamW

from vilbert.optimization import RAdam
from vilbert_ipu import (
    task_utils_ipu,
    ipu_options,
    RetrievalFlickr30k,
)
from vilbert.vilbert import BertConfig

import vilbert.utils as utils
# import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
opts = ipu_options.opts

def main():

    args = task_utils_ipu.GetParser().parse_args()
    with open("vilbert_tasks.yml", "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    opts.randomSeed(args.seed)

    task_id = "TASK" + args.tasks
    task_name = task_cfg[task_id]["name"] 
    base_lr = task_cfg[task_id]["lr"]

    timeStamp = (task_name + "_" + args.config_file.split("/")[1].split(".")[0] + "-" + args.save_name)
    savePath = os.path.join(args.output_dir, timeStamp)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    config = BertConfig.from_json_file(args.config_file)

    with open(os.path.join(savePath, "command.txt"), "w") as f:
        print(args, file=f)  # Python 3.x
        print("\n", file=f)
        print(config, file=f)

    # use_fake_data 
    print("use_fake_data: ", args.use_fake_data)
    if args.use_fake_data:
        
        task_batch_size, task_num_iters, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val = task_utils_ipu.GenFakeDatasets(
            args, task_cfg, [task_id], opts
        )
        num_labels = 1
    else:
        task_batch_size, task_num_iters, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val = task_utils_ipu.LoadDatasets(
            args, task_cfg, [task_id], opts
        )
        num_labels = max([dataset.num_labels for dataset in task_datasets_train.values()])
        print("num_labels", num_labels)
    # only single task
    task_dataloader_train=task_dataloader_train[task_id]

    logdir = os.path.join(savePath, "logs")
    tbLogger = utils.tbLogger(
        logdir,
        savePath,
        task_name,
        [task_id],
        task_num_iters,
        args.gradient_accumulation_steps,
    )

    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target

    if args.task_specific_tokens:
        config.task_specific_tokens = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    median_num_iter = int(
        task_cfg[task_id]["num_epoch"]
        * task_num_iters[task_id]
        * args.train_iter_multiplier
        / args.num_train_epochs
    )
    print("median_num_iter", median_num_iter)
    task_stop_controller = utils.MultiTaskStopOnPlateau(
        mode="max",
        patience=1,
        continue_threshold=0.005,
        cooldown=1,
        threshold=0.001,
    )

    

    if args.dynamic_attention:
        config.dynamic_attention = True
    if "roberta" in args.bert_model:
        config.model = "roberta"
    
    model = RetrievalFlickr30k.PipelinedWithLossForRetrievalFlickr30k(
        config=config,
        args = args,
        num_labels=num_labels
    )
    # model = model.half()

    optimizer, warmup_scheduler, lr_scheduler, lr_reduce_list, warmpu_steps = task_utils_ipu.GetOptimizer(
            args, model, base_lr, median_num_iter
        )

    startIterID = 0
    global_step = 0
    start_epoch = 0

    if args.resume_file != "" and os.path.exists(args.resume_file):
        checkpoint = torch.load(args.resume_file, map_location="cpu")
        new_dict = {}
        for attr in checkpoint["model_state_dict"]:
            if attr.startswith("module."):
                new_dict[attr.replace("module.", "", 1)] = checkpoint[
                    "model_state_dict"
                ][attr]
            else:
                new_dict[attr] = checkpoint["model_state_dict"][attr]
        model.load_state_dict(new_dict)
        warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        start_epoch = int(checkpoint["epoch_id"]) + 1
        task_stop_controller = checkpoint["task_stop_controller"]
        tbLogger = checkpoint["tb_logger"]
        del checkpoint


    # if default_gpu:
    print("***** Running training *****")
    print("  Num Iters: ", task_num_iters)
    print("  Batch size: ", task_batch_size)
    

    task_iter_train = None
    task_count = 0
    
    
    # # # # # # # # #   
    #  start train  #
    # # # # # # # # #

    # for testing, you can use isIPU to change how to run this code in IPU or CPU
    isIPU = args.enable_IPU
    print('enable_IPU: ', isIPU)

    train_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)
    # inference_model = poptorch.inferenceModel(model, options=opts)

    for epochId in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"):
        
        torch.autograd.set_detect_anomaly(True)
        for step in range(median_num_iter):
            iterId = startIterID + step + (epochId * median_num_iter)
            
            model.train()
            is_forward = False
            if (not task_stop_controller.in_stop) or (
                iterId % args.train_iter_gap == 0
            ):
                is_forward = True
            # given the current task, decided whether to forward the model and forward with specific loss.

            # reset the task iteration when needed.
            

            
            
            if args.use_fake_data:
                batch = task_dataloader_train.next()
            else:
                if task_count % len(task_dataloader_train) == 0:
                    task_iter_train = iter(task_dataloader_train)
                batch = tuple(task_iter_train.next()) # get the batch
            
            task_count += 1

            if is_forward:  
                
                if isIPU:
                    score, loss = train_model(batch) 
                    # IPU will auto backforward
                else:
                    score, loss = model(batch)   #  test in CPU first to make sure there is no error in model
                    loss.backward() 
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if not isIPU:
                        optimizer.step() 
                    # model.zero_grad()
                    if global_step < warmpu_steps or args.lr_scheduler == "warmup_linear":
                        warmup_scheduler.step()
                        train_model.setOptimizer(optimizer)           
                    global_step += 1

                    tbLogger.step_train(
                        epochId,
                        iterId,
                        float(loss),
                        float(score),
                        optimizer.param_groups[0]["lr"],
                        task_id,
                        "train",
                    )

            if "cosine" in args.lr_scheduler and global_step > warmpu_steps:
                lr_scheduler.step()
                train_model.setOptimizer(optimizer)
            if (
                step % (20 * args.gradient_accumulation_steps) == 0
                and step != 0
                # and default_gpu
            ):
                tbLogger.showLossTrain()

            # decided whether to evaluate on each tasks.
            # if (iterId != 0 and iterId % task_num_iters[task_id] == 0) or (
            #     epochId == args.num_train_epochs - 1 and step == median_num_iter - 1
            # ):
            #     model.eval()
            #     for i, batch in enumerate(task_dataloader_val[task_id]):
            #         if isIPU:
            #             _, batch_size, _, loss = inference_model(batch)
            #         else:
            #             _, batch_size, _, loss = model(batch) # test in CPU
            #         tbLogger.step_val(
            #             epochId, float(loss), float(score), task_id, batch_size, "val"
            #         )
            #         # if default_gpu:
            #         sys.stdout.write("%d/%d\r" % (i, len(task_dataloader_val[task_id])))
            #         sys.stdout.flush()

            #     # update the multi-task scheduler.
            #     task_stop_controller.step(tbLogger.getValScore(task_id))
            #     score = tbLogger.showLossVal(task_id, task_stop_controller)
            #     model.train()

        if args.lr_scheduler == "automatic":
            lr_scheduler.step(sum(tbLogger.showLossValAll().values()))
            train_model.setOptimizer(optimizer)
            logger.info("best average score is %3f" % lr_scheduler.best)
        elif args.lr_scheduler == "mannul":
            lr_scheduler.step()
            train_model.setOptimizer(optimizer)

        if epochId in lr_reduce_list:
            task_stop_controller._reset()

        # Save a trained model
        logger.info("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Only save the model it-self
        output_model_file = os.path.join(
            savePath, "pytorch_model_" + str(epochId) + ".bin"
        )
        output_checkpoint = os.path.join(savePath, "pytorch_ckpt_latest.tar")
        torch.save(model_to_save.state_dict(), output_model_file)
        torch.save(
            {
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                # 'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                "global_step": global_step,
                "epoch_id": epochId,
                "task_stop_controller": task_stop_controller,
                "tb_logger": tbLogger,
            },
            output_checkpoint,
        )
    tbLogger.txt_close()




if __name__ == "__main__":

    main()
