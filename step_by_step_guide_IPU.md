# creat vituel env by conda. 

note: it is OK if you have other IPU env created by virtualenv or something. This part for who begin a new IPU env.

```
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

touch $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
touch $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

you can add any environment variables if you want into **activate.d/env_vars.sh** .

And unset those environment variables into **deactivate.d/env_vars.sh** .

For this project, you need to add IPU libs into environment variables to enable IPU settings.

```
cat>$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh<<EOF
#!/bin/bash
export GCDA_MONITOR=1
export TF_CPP_VMODULE="poplar_compiler=1"
export TF_POPLAR_FLAGS="--max_compilation_threads=40 --executable_cache_path=/<your_cachedir>"
export TMPDIR="/<your_tmpdir>"
export POPART_LOG_LEVEL=DEBUG
export POPTORCH_LOG_LEVEL=DEBUG_IR
export POPLAR_LOG_LEVEL=DEBUG
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./reports"}'

source /<your_popart_path>/enable.sh
source /<your_poplar_path>/enable.sh
EOF
```
then unset Poplar SDK in deactivate.d
```
cat>$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh<<EOF
#!/bin/bash
unset POPLAR_SDK_ENABLED
EOF
```

Using this virtuel env by conda activate command

``` 
conda activate vilbert-mt 
```
# build project

select a dir as your workspace for this project
```
git clone https://github.com/KAndHisC/vilbert-multi-task.git
git checkout IPU

# install libcap-dev. Example in ubuntu is following.
apt-get install build-essential libcap-dev

# install some dependencies
conda install  torchvision torchaudio cpuonly -c pytorch
conda install jupyter ninja cython matplotlib pandas h5py tqdm
conda install -c conda-forge scikit-image
pip install -r requirements.txt

# in the end, install poptorch/torch/tensorflow2 from whl in poplar SDK --
```
install refer
```
cd vilbert-multi-task/tools/
git clone https://github.com/lichengunc/refer.git
cd refer
git checkout python3
make
python setup.py install
```

back to dir of vilbert-multi-task and install it in develop mode

```
cd vilbert-multi-task/
python setup.py build develop
```
get all config
```
# cd vilbert-multi-task/data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```
get models, you can put the model in other places if you want
```
# cd vilbert-multi-task/
# mkdir save
# cd save 
# mkdir origin
# cd origin
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin
```
get datasets 
```
cd vilbert-multi-task/data/
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets.tar.gz 
tar xf datasets.tar.gz
```
<!-- you can run a fintune task by following command
```
python train_tasks_ipu.py --bert_model bert-base-uncased --from_pretrained /localdata/takiw/vilbert/save/origin/pretrained_model.bin --output_dir /localdata/takiw/vilbert/save --config_file config/bert_base_6layer_6conect.json --tasks 8 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name flickr30k_finetune_copy 
``` -->

you can run a example by this bash script `run_retrevalFlicker30K_task.sh` .

```
bash run_retrevalFlicker30K_task.sh
bash run_retrevalFlicker30K_task.sh --enable_IPU
bash run_retrevalFlicker30K_task.sh --enable_IPU --use_fake_data
nohup bash run_retrevalFlicker30K_task.sh --enable_IPU --use_fake_data > train.log 2>&1 &
```
the code for training finetune of RetrievalFlickr30k is a simplified version of multi-tasks training which can help understand ViL BERT easier. The training loop is in vilbert-multi-task/train_retrieval_ipu.py.

And the pipelinedModel is in vilbert-multi-task/vilbert_ipu/RetrievalFlickr30k.py


31 08
31 56

12
59
42
100*28

48

100 * 21

43.75
