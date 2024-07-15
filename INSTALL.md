# 🖥️ Setup

This code was tested on Ubuntu 22.04 LTS, RTX A5000, CUDA 12.1.

Clone the repository:

```bash
git clone https://github.com/AndreeaDogaru/Gen3DSR
cd Gen3DSR
EXT_DIR=$(pwd)/external
```
For a successful installation, the locally installed CUDA toolkit should match with the one in the PyTorch binaries. If CUDA 12.1 is not available, we recommend downloading the corresponding runfile from [here](https://developer.nvidia.com/cuda-12-1-0-download-archive) and installing the toolkit:
```bash
./runfile.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.1/
```

Set environment variables to avoid CUDA conflicts:

```bash
export CUDA_HOME=/usr/local/cuda-12.1/
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64"  
export LIBRARY_PATH=$CUDA_HOME/lib64  
export LD_LIBRARY_PATH=$CUDA_HOME/lib64  
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"  
```

It is recommended to use a virtual environment with Python 3.10. Install requirements using pip:
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Install additional dependencies:
```bash
pip install git+https://github.com/jinlinyi/PerspectiveFields.git
pip install git+https://github.com/NVlabs/nvdiffrast
pip install git+https://github.com/ashawkey/kiuikit 
pip install https://github.com/AndreeaDogaru/mmcv/releases/download/v2.1.0/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl
pip install $EXT_DIR/dreamgaussian/diff-gaussian-rasterization 
pip install $EXT_DIR/dreamgaussian/simple-knn 
python3 -m pip install -v -U xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/open-mmlab/mmdetection.git 
pip install natten==0.15.1+torch220cu121 -f https://shi-labs.com/natten/wheels/ 
python -m pip install -e $EXT_DIR/detectron2 

cd $EXT_DIR/detectron2/projects/CropFormer/entity_api/PythonAPI
python3 setup.py build_ext --inplace
cd $EXT_DIR/detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops
python3 setup.py build install
```

You can download the checkpoints either manually by placing them in the specified directory or automatically using the included [script](external/checkpoints/download.sh). 
Optional: as all the checkpoints require about 30GB of local storage, you can selectively only download the ones for the models you plan to use. For the model provided by [Adobe_EntitySeg](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg), you must first accept the repository's terms and then use a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) to download it.
```bash
cd $EXT_DIR/checkpoints
./download.sh <HF_TOKEN>
cd $EXT_DIR/..
```

