# export POPART_LOG_CONFIG=/localdata/takiw/workspace/vilbert-multi-task/conf.py
# echo $POPART_LOG_CONFIG
python train_retrieval_ipu.py $1 $2 --bert_model bert-base-uncased --from_pretrained save/origin/pretrained_model.bin --output_dir save --config_file config/bert_base_6layer_6conect.json --tasks 8 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name flickr30k_finetune_copy 
