
# set the workspace
export INSTALL_DIR=$PWD

# creat vituel env
cd $INSTALL_DIR
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
cd vilbert-multi-task

# install some dependencies
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install jupyter ninja cython matplotlib pandas h5py
conda install -c conda-forge scikit-image
pip install -r requirements.txt
pip install yacs

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex
cd apex
source /fsx/sw_pkgs/envs.cuda.10.0 # in our aws default path referce to CUDA 9, need changed to 10.0
python setup.py install --cuda_ext --cpp_ext
pip install -v --disable-pip-version-check --use-feature=in-tree-build --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
conda install tqdm
pip install opencv-python # already haved 
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install PyTorch Detection
<!-- cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop 
# now it was replaced by vqa-maskrcnn-benchmark.git
# because the repo maskrcnn-benchmark updated to a high version and miss matched this project.
-->

# install vqa-maskrcnn-benchmark
cd $INSTALL_DIR/
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
python setup.py build develop

# install refer
cd $INSTALL_DIR/vilbert-multi-task/tools/
git clone https://github.com/lichengunc/refer.git
cd refer
git checkout python3
make

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


# to get raw dataset(example:Flickr30k) from kaggle
pip install kaggle
cd ~
mkdir .kaggle
cd ~/.kaggle/
touch kaggle.json # then put your kaggle-api-token in this file
cd cd INSTALL_DIR/vilbert-multi-task/data/
mkdir rawdatasls
ets
cd rawdatasets
kaggle datasets download -d hsankesara/flickr-image-dataset
unzip flickr-image-dataset.zip
#### flickr dataset have a wrong in packing, there are two copies in this zip, you can delete one of them.

# extract feature from rawdata
cd $INSTALL_DIR/vilbert-multi-task/
python script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir data/rawdatasets/flickr30k/flickr30k_images --output_folder data/rawdatasets/flickr30k/image_features

# write into lmdb
python script/convert_to_lmdb.py --features_dir data/rawdatasets/flickr30k/image_features --lmdb_file data/rawdatasets/flickr30k/flickr30k.lmdb

# add a new task in vilbert_tasks.yml
# add this new task in vilbert/dataset/__init__.py
# run train task 

nohup python train_tasks.py --bert_model bert-base-uncased --from_pretrained models/pretrained_model.bin --config_file config/bert_base_6layer_6conect.json --tasks 19 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name flickr30k_finetune_copy >> out.log 2>&1 &






