#!/bin/sh
pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
pip3 install -U opencv-python
pip3 install git+https://github.com/cocodataset/panopticapi.git
pip3 install git+https://github.com/mcordts/cityscapesScripts.git