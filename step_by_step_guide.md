
# set the workspace
export INSTALL_DIR=$PWD

# creat vituel env
cd $INSTALL_DIR
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
cd vilbert-multi-task
pip install -r requirements.txt

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install h5py

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex
cd apex
source /fsx/sw_pkgs/envs.cuda.10.0 # in our aws default path referce to CUDA 9, need changed to 10.0
python setup.py install --cuda_ext --cpp_ext
pip install -v --disable-pip-version-check --use-feature=in-tree-build --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


# install vqa-maskrcnn-benchmark
cd $INSTALL_DIR
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
conda install jupyter ninja cython matplotlib
pip install yacs
conda install torchvision -c pytorch # aleady haved

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
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

# install refer
git clone https://github.com/lichengunc/refer.git
cd refer
make

# install vilbert-multi-task
cd $INSTALL_DIR/vilbert-multi-task/
python setup.py develop










