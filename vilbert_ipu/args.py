# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import yaml
import argparse


config_file = "./configs.yml"


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def parse_args(args=None):
    pparser = argparse.ArgumentParser("ViT Configuration name", add_help=False)
    pparser.add_argument("--config",
                         type=str,
                         help="Configuration Name",
                         default='vil_bert')
    pargs, remaining_args = pparser.parse_known_args(args=args)
    config_name = pargs.config

    parser = argparse.ArgumentParser(
        "Poptorch ViT",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Execution
    parser.add_argument("--batch-size", type=int, help="Set the micro batch-size")
    parser.add_argument("--training-steps", type=int, help="Number of training steps")
    parser.add_argument("--batches-per-step", type=int, help="Number of batches per training step")
    parser.add_argument("--replication-factor", type=int, help="Number of replicas")
    parser.add_argument("--pred-head-transform", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable prediction head transform in the CLS layer during pretraining. This transform comes after\
                        the encoders but before the decoder projection for the MLM loss.")
    parser.add_argument("--gradient-accumulation", type=int, help="Number of gradients to accumulate before updating the weights")
    parser.add_argument("--embedding-serialization-factor", type=int, help="Matmul serialization factor the embedding layers")
    parser.add_argument("--recompute-checkpoint-every-layer", type=str_to_bool, nargs="?", const=True, default=False,
                        help="This controls how recomputation is handled in pipelining. "
                        "If True the output of each encoder layer will be stashed keeping the max liveness "
                        "of activations to be at most one layer. "
                        "However, the stash size scales with the number of pipeline stages so this may not always be beneficial. "
                        "The added stash + code could be greater than the reduction in temporary memory.",)
    parser.add_argument("--enable-half-partials", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable half partials for matmuls and convolutions globally")
    parser.add_argument("--ipus-per-replica", type=int, help="Number of IPUs required by each replica")
    parser.add_argument("--encoder-start-ipu", type=int, choices=[0, 1],
                        help="The index of the IPU that the first encoder will be placed on. Can be 0 or 1.")
    parser.add_argument("--layers-per-ipu", type=int, nargs="+",
                        help="Number of encoders placed on each IPU. Can be a single number, for an equal number encoder layers per IPU.\
                              Or it can be a list of numbers, specifying number of encoder layers for each individual IPU.")
    parser.add_argument("--matmul-proportion", type=float, nargs="+", help="Relative IPU memory proportion size allocated for matmul")
    parser.add_argument("--async-dataloader", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable asynchronous mode in the DataLoader")
    parser.add_argument("--file-buffer-size", type=int, help="Number of files to load into the Dataset internal buffer for shuffling.")
    parser.add_argument("--random-seed", type=int, help="Seed for RNG")
    parser.add_argument('--precision', choices=['16.16', '16.32', '32.32'], default='16.16', help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 16.32, 32.32")
    parser.add_argument('--normalization-location', choices=['host', 'ipu', 'none'], default='host', help='Location of the data normalization')
    parser.add_argument("--layers-on-ipu", type=float, nargs="+", help="Relative IPU memory proportion size allocated for matmul")

    # Optimizer
    parser.add_argument("--optimizer", type=str, choices=['AdamW', 'Adam', 'SGD', 'LAMB', 'LAMBNoBias'], help="optimizer to use for the training")
    parser.add_argument("--learning-rate", type=float, help="Learning rate value for constant schedule, maximum for linear schedule.")
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear"],
                        help="Type of learning rate schedule. --learning-rate will be used as the max value")
    parser.add_argument("--lr-warmup", type=float, help="Proportion of lr-schedule spent in warm-up. Number in range [0.0, 1.0]")
    parser.add_argument("--loss-scaling", type=float, help="Loss scaling factor (recommend using powers of 2)")
    parser.add_argument("--weight-decay", type=float, help="Set the weight decay")
    parser.add_argument("--momentum", type=float, help="The momentum factor of SGD optimizer")
    parser.add_argument("--warmup-steps", type=int, help="The momentum factor of SGD optimizer")


    # Model
    parser.add_argument("--sequence-length", type=int, help="The max sequence length")
    parser.add_argument("--mask-tokens", type=int, help="Set the max number of MLM tokens in the input dataset.")
    parser.add_argument("--vocab-size", type=int, help="Set the size of the vocabulary")
    parser.add_argument("--hidden-size", type=int, help="The size of the hidden state of the transformer layers")
    parser.add_argument("--intermediate-size", type=int, help="hidden-size*4")
    parser.add_argument("--num-hidden-layers", type=int, help="The number of transformer layers")
    parser.add_argument("--num-attention-heads", type=int, help="Set the number of heads in self attention")
    parser.add_argument("--layer-norm-eps", type=float, help="The eps value for the layer norms")
    parser.add_argument("--use-small-embedding-size", type=str_to_bool, nargs="?", const=True, default=False, help="The eps value for the layer norms")
    parser.add_argument("--embedding-size", type=int, help="The eps value for the layer norms")
    parser.add_argument("--mlp-dim", type=int, help="The size of mlp dimention")
    parser.add_argument("--dropout-prob", type=float, nargs="?", const=True, help="Cls dropout probability")
    parser.add_argument("--patches-size", type=float, nargs="+", help="The size of image tokens")
    parser.add_argument("--num-classes", type=int, help="The classes of image")

    # Hugging Face specific
    parser.add_argument("--attention-probs-dropout-prob", type=float, nargs="?", const=True, help="Attention dropout probability")

    # Dataset
    parser.add_argument('--dataset', choices=['cifar10', 'imagenet', 'synthetic', 'generated'], default='cifar10', help="Choose data")
    parser.add_argument("--input-files", type=str, nargs="+", help="Input data files")
    parser.add_argument("--synthetic-data", type=str_to_bool, nargs="?", const=True, default=False,
                        help="No Host/IPU I/O, random data created on device")

    # Misc
    parser.add_argument("--dataloader-workers", type=int, help="The number of dataloader workers")
    parser.add_argument("--profile", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable profiling")
    parser.add_argument("--profile-dir", type=str, help="Directory for profiling results")
    parser.add_argument("--custom-ops", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable custom ops")
    parser.add_argument("--wandb", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling logging to Weights and Biases")
    parser.add_argument("--use-popdist", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling poprun function")
    parser.add_argument("--popdist-size", type=int, help="The popdist size")
    parser.add_argument("--enable-rts", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling RTS")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Directory where checkpoints will be saved and restored from.\
                             This can be either an absolute or relative path. If this is not specified, only end of run checkpoint is\
                             saved in an automatically generated directory at the root of this project. Specifying directory is\
                             recommended to keep track of checkpoints.")
    parser.add_argument("--checkpoint-every-epoch", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Option to checkpoint model after each epoch.")
    parser.add_argument("--checkpoint-save-steps", type=int, default=100,
                        help="Option to checkpoint model after n steps.")
    parser.add_argument("--restore-epochs-and-optimizer", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Restore epoch and optimizer state to continue training. This should normally be True when resuming a\
                              previously stopped run, otherwise False.")
    parser.add_argument("--checkpoint-file", type=str, default="", help="Checkpoint to be retrieved for further training. This can\
                              be either an absolute or relative path to the checkpoint file.")
    parser.add_argument("--restore", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Restore a checkpoint model to continue training.")

    # This is here only for the help message
    parser.add_argument("--config", type=str, help="Configuration name")

    # VilBERT
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
    parser.add_argument( "--num_workers", type=int, default=8, help="Number of workers in the dataloader.")
    parser.add_argument( "--save_name", default="", type=str, help="save name for training.")
    parser.add_argument("--tasks", default='8', type=str, help="1-2-3... training task separate by -")
    parser.add_argument("--freeze", default=-1, type=int, help="till which layer of textual stream of vilbert need to fixed.")
    parser.add_argument("--vision_scratch", action="store_true", help="whether pre-trained the image or not.")
    parser.add_argument("--evaluation_interval", default=1, type=int, help="evaluate very n epoch.")
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

    # Load the yaml
    yaml_args = dict()
    if config_name is not None:
        with open(config_file, "r") as f:
            try:
                yaml_args.update(**yaml.safe_load(f)[config_name])
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    # Check the yaml args are valid
    known_args = set(vars(parser.parse_args("")))
    unknown_args = set(yaml_args) - known_args

    if unknown_args:
        print(f" Warning: Unknown arg(s) in config file: {unknown_args}")

    parser.set_defaults(**yaml_args)
    args = parser.parse_args(remaining_args)

    # Expand layers_per_ipu input into list representation
    if isinstance(args.layers_per_ipu, int):
        args.layers_per_ipu = [args.layers_per_ipu]

    if len(args.layers_per_ipu) == 1:
        layers_per_ipu_ = args.layers_per_ipu[0]
        args.layers_per_ipu = [layers_per_ipu_] * (args.num_hidden_layers // layers_per_ipu_)

    if sum(args.layers_per_ipu) != args.num_hidden_layers:
        raise ValueError(f"layers_per_ipu not compatible with number of hidden layers: {args.layers_per_ipu} and {args.num_hidden_layers}")

    # Expand matmul_proportion input into list representation
    if isinstance(args.matmul_proportion, float):
        args.matmul_proportion = [args.matmul_proportion] * args.ipus_per_replica

    if len(args.matmul_proportion) != args.ipus_per_replica:
        if len(args.matmul_proportion) == 1:
            args.matmul_proportion = args.matmul_proportion * args.ipus_per_replica
        else:
            raise ValueError(f"Length of matmul_proportion doesn't match ipus_per_replica: {args.matmul_proportion} vs {args.ipus_per_replica}")

    args.global_batch_size = args.replication_factor * args.gradient_accumulation * args.batch_size
    args.samples_per_step = args.global_batch_size * args.batches_per_step
    args.intermediate_size = args.hidden_size * 4
    return args
