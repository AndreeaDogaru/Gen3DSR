# Gen3DSR: Generalizable 3D Scene Reconstruction via Divide and Conquer from a Single View

This repository contains the official implementation of the 3DV2025 paper titled *Generalizable 3D Scene Reconstruction via Divide and Conquer from a Single View*.

[![Website](https://img.shields.io/badge/Project-Page-b361ff
)](https://andreeadogaru.github.io/Gen3DSR/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2404.03421)

[Andreea Ardelean](https://andreeadogaru.github.io/),
[Mert Özer](),
[Bernhard Egger](https://eggerbernhard.ch/)

![results](teaser.png)

## 📢 News
21.03.2025: Evaluation code is released. <br>
06.11.2024: Gen3DSR is accepted at 3DV2025! <br>
15.07.2024: Inference code is released. <br>
04.04.2024: Paper is available on <a href="https://arxiv.org/pdf/2404.03421"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" height="16"></a>. 

## 🎮 Usage
- Interactive scene reconstruction examples can be found on our [project page](https://andreeadogaru.github.io/Gen3DSR/). 
- A Dockerfile is provided for easy setup:
```bash
docker build --build-arg HF_TOKEN=<USER_TOKEN> -t gen3dsr .
docker run -it --gpus all \
    --mount type=bind,source="$(pwd)/src",target=/app/src \
    --mount type=bind,source="$(pwd)/imgs",target=/app/imgs \
    --mount type=bind,source="$(pwd)/out",target=/app/out \
    gen3dsr python run.py --config ./configs/image.yaml
```
- Alternatively, after following the detailed local installation instructions presented in [INSTALL.md](INSTALL.md), the code can be run as follows:
```bash
cd src
python run.py --config ./configs/image.yaml \
    scene.attributes.img_path='../imgs/demo_1.jpg' \
    scene.save_dir='../out/demo_1'
```
- We recommend the user to check out the different inference options in [image.yaml](src/configs/image.yaml). 
- The reconstruction can be found in `<scene.save_dir>/reconstruction`.
- The code requires minimum 20GB of VRAM.

## 📋 Evaluation
We provide here the information required to evaluate Gen3DSR on the subset of 100 scenes selected from the [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) dataset as in our paper. 
- Downloading the data:
```bash
cd imgs
git lfs install
git clone https://huggingface.co/datasets/andreead-a/FRONT3D
```
- Running the method on a given scene: 
```bash
cd src
python run.py --config ./configs/front.yaml scene.attributes.render_task='5131' 
```
- The reconstruction can be found in `out/front3d/rec_<scene.attributes.render_task>`
- A script is provided to evaluate all reconstructed scenes against the ground truth:
```bash
cd src
python eval_front.py --data_root ../imgs/FRONT3D --rec_path ../out/front3d
```
- The results will be saved at `out/front3d/metrics_full.json`
## 🎓 Citation
```
@inproceedings{Ardelean2025Gen3DSR,
    title={Generalizable 3D Scene Reconstruction via Divide and Conquer from a Single View},
    author={Ardelean, Andreea and Özer, Mert and Egger, Bernhard},
    booktitle = {International Conference on 3D Vision (3DV)},
    year={2025}
}
```

## 🙌 Acknowledgements

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://github.com/dreamgaussian/dreamgaussian)
- [High Quality Entity Segmentation](https://github.com/qqlu/Entity/blob/main/Entityv2)
- [Perspective Fields for Single Image Camera Calibration](https://github.com/jinlinyi/PerspectiveFields)
- [Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://github.com/prs-eth/Marigold)
- [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://github.com/LiheYoung/Depth-Anything)
- [Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image](https://github.com/YvanYin/Metric3D)
- [OneFormer: One Transformer to Rule Universal Image Segmentation](https://github.com/SHI-Labs/OneFormer)
- [One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization](https://github.com/One-2-3-45/One-2-3-45)
- [Open-Vocabulary SAM: Segment and Recognize Twenty-thousand Classes Interactively](https://github.com/HarborYuan/ovsam)

This work was funded by the German Federal Ministry of Education and Research (BMBF), FKZ: 01IS22082 (IRRW). The authors are responsible for the content of this publication. The authors gratefully acknowledge the scientific support and HPC resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) under the NHR project b112dc IRRW. NHR funding is provided by federal and Bavarian state authorities. NHR@FAU hardware is partially funded by the German Research Foundation (DFG) – 440719683.

## 📝 License

The code is released under the CC BY 4.0 [LICENSE](LICENSE).

