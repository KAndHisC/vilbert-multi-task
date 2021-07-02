
# set the workspace
export INSTALL_DIR=$PWD

# creat vituel env
cd $INSTALL_DIR
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

touch $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
touch $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

#### you can add any environment variables if you want into activate.d/env_vars.sh
#### unset those environment variables into deactivate.d/env_vars.sh

# example, add IPU libs into environment variables
cat>$CONDA_PREFIX//etc/conda/activate.d/env_vars.sh<<EOF
#!/bin/bash
export GCDA_MONITOR=1
export TF_CPP_VMODULE="poplar_compiler=1"
export TF_POPLAR_FLAGS="--max_compilation_threads=40 --executable_cache_path=/<your_cachedir>"
export TMPDIR="/<your_tmpdir>"
export POPLAR_LOG_LEVEL=INFO

source /<your_popart_path>/enable.sh
source /<your_poplar_path>/enable.sh
EOF

cat>$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh<<EOF
#!/bin/bash
unset POPLAR_SDK_ENABLED
EOF

# activate virtuel env again
conda activate vilbert-mt

# get project
git clone https://github.com/KAndHisC/vilbert-multi-task.git
cd vilbert-multi-task

# install libcap-dev. Example in ubuntu is following.
apt-get install build-essential libcap-dev
# install some dependencies
#### install poptorch/tensorflow2 from whl in SDK --
conda install  torchvision torchaudio cpuonly -c pytorch
conda install jupyter ninja cython matplotlib pandas h5py tqdm
conda install -c conda-forge scikit-image
pip install -r requirements.txt

<!-- # install apex  
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex
cd apex
source /fsx/sw_pkgs/envs.cuda.10.0 # in our aws default path referce to CUDA 9, need changed to 10.0
python setup.py install --cuda_ext --cpp_ext
pip install -v --disable-pip-version-check --use-feature=in-tree-build --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ -->
# Half and mixed precision is supported in PopTorch.

# after this line are needed.
# install refer
cd $INSTALL_DIR/vilbert-multi-task/tools/
git clone https://github.com/lichengunc/refer.git
cd refer
git checkout python3
make
python setup.py install

# install vilbert-multi-task
cd $INSTALL_DIR/vilbert-multi-task/
python setup.py build develop

# the config files for extracting features
cd INSTALL_DIR/vilbert-multi-task/data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml

# to get pretrain datasets 
cd INSTALL_DIR/vilbert-multi-task/data/
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets.tar.gz 
tar xf datasets.tar.gz


# add a new task in vilbert_tasks.yml
# add this new task in vilbert/dataset/__init__.py
# run train task 

nohup python train_tasks_ipu.py --bert_model bert-base-uncased --from_pretrained /localdata/takiw/vilbert/save/origin/pretrained_model.bin --output_dir /localdata/takiw/vilbert/save --config_file config/bert_base_6layer_6conect.json --tasks 8 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name flickr30k_finetune_copy >> out.log 2>&1 &



nohup python train_retrieval_ipu.py --bert_model bert-base-uncased --from_pretrained save/origin/pretrained_model.bin --output_dir save --config_file config/bert_base_6layer_6conect.json --tasks 8 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name flickr30k_finetune_copy >> out.log 2>&1 &




