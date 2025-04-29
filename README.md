<div align="center">

# [GaussTR](): Foundation Model-Aligned [Gauss]()ian [Tr]()ansformer for Self-Supervised 3D Spatial Understanding

[Haoyi Jiang](https://scholar.google.com/citations?user=_45BVtQAAAAJ)<sup>1</sup>, Liu Liu<sup>2</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ)<sup>1</sup>, Xinjie Wang<sup>2</sup>,
[Tianwei Lin](https://wzmsltw.github.io/)<sup>2</sup>, Zhizhong Su<sup>2</sup>, Wenyu Liu<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1</sup><br>
<sup>1</sup>Huazhong University of Science & Technology, <sup>2</sup>Horizon Robotics

[**CVPR 2025**]()

[![Project page](https://img.shields.io/badge/project%20page-hustvl.github.io%2FGaussTR-blue)](https://hustvl.github.io/GaussTR/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.13193-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2412.13193)
[![License: MIT](https://img.shields.io/github/license/hustvl/GaussTR)](LICENSE)

</div>

## News

* ***Feb 27 '25:*** Our paper has been accepted at CVPR 2025. 🎉
* ***Feb 11 '25:*** Released the model integrated with Talk2DINO, achieving new state-of-the-art results.
* ***Dec 17 '24:*** Released our arXiv paper along with the source code.

## Setup

### Installation

We recommend cloning the repository using the `--single-branch` option to avoid downloading unnecessary large media files for the project website from other branches:

```bash
git clone git@github.com:JunShao0104/GaussTR.git
# Create your own branch and switch to it
git checkout -b your_branch_name
git push -u origin your_branch_name
cd GaussTR
conda create -n gausstr python=3.10

# For CUDA 12.6
pip install -r requirements.txt
pip uninstall mmcv==2.2.0
pip install mmcv==2.1.0

# For CUDA 11.8
pip install -r requirements_cuda118.txt --extra-index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install -r requirements_cuda121.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Install gsplat for CUDA 11.8
# https://docs.gsplat.studio/whl/gsplat/
cd whl/
wget https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt21cu118-cp310-cp310-linux_x86_64.whl
pip install gsplat-1.4.0+pt21cu118-cp310-cp310-linux_x86_64.whl
cd ..

# To install mmdet3d development version in order to generate the pkl files and gt database:
# pip uninstall mmdet3d
# wget https://github.com/open-mmlab/mmdetection3d/archive/refs/tags/v1.4.0.zip
# unzip v1.4.0.zip
# cd mmdetection3d-1.4.0
# pip install -v -e .

# To generate the train and val pkl files:
# Create symbolic links to the dataset: ./data/nuscenes under mmdetection3d-1.4.0
# cd mmdetection3d-1.4.0
# python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes # train and val pkl files, db infos pkl file, gt_database
# python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --only-gt-database # only gt_database

# Fore reference, the author's packages version
# Package                 Version            Editable project location
# ----------------------- ------------------ ---------------------------------------------------
# gsplat                  1.4.0+pt21cu118
# mmcv                    2.1.0
# mmdet                   3.3.0
# mmdet3d                 1.4.0
# mmengine                0.10.4
# mmpretrain              1.2.0
# mmsegmentation          1.2.2
# numpy                   1.26.4
# openmim                 0.3.9
# torch                   2.1.2+cu118
# torchvision             0.16.2+cu118
```

### Dataset Preparation

1. Prepare the nuScenes dataset following the instructions in the [mmdetection3d docs](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html#nuscenes).
2. Update the dataset `.pkl` files with `scene_idx` to match the occupancy ground truths:

    ```bash
    python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
    ```

3. Download the occupancy ground truth data from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and place it in `data/nuscenes/gts`.
4. Generate features and rendering targets:

    * Run `PYTHONPATH=. python tools/generate_depth.py` to generate metric depth estimations.
    * **[For GaussTR-FeatUp Only]** Navigate to the [FeatUp](https://github.com/mhamilton723/FeatUp) repository and run `python tools/generate_featup.py`.
    * **[Optional for GaussTR-FeatUp]** Navigate to the [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) and run `python tools/generate_grounded_sam2.py` to enable auxiliary segmentation supervision.

```bash
# Error list and solutions:
# 1. For RuntimeError: Failed to find function: mono.model.backbones.vit_large_reg
# https://github.com/YvanYin/Metric3D/issues/151
# Navigate to /home/lzhao360/.cache/torch/hub/yvanyin_metric3d_main/mono/utils/comm.py and add the following line:
from mono.model.backbones import *

# 2. For ModuleNotFoundError: No module named 'mmcv._ext'
# https://mmcv.readthedocs.io/en/latest/get_started/installation.html
pip uninstall mmcv mmcv-full
# Replace the cu118 and torch2.1 with your own CUDA and PyTorch version
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
# Replace the cu121 and torch2.3 with your own CUDA and PyTorch version
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html

# 3. For FileNotFoundError: [Errno 2] No such file or directory: '/scratch/lzhao360/download/miniconda3/miniconda3/envs/gausstr/lib/python3.10/site-packages/featup/featurizers/maskclip/bpe_simple_vocab_16e6.txt.gz'
# https://github.com/mhamilton723/FeatUp/issues/47
cd /scratch/lzhao360/download/miniconda3/miniconda3/envs/gausstr/lib/python3.10/site-packages/featup/featurizers/maskclip/
wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz -O bpe_simple_vocab_16e6.txt.gz
gzip -t bpe_simple_vocab_16e6.txt.gz # For checking if the file is valid

# 4. For ModuleNotFoundError: No module named 'supervision' / 'transformers'
pip install supervision transformers

# 5. For ModuleNotFoundError: No module named 'grounding_dino'
# Search all the grounding_dino.groundingdino in the Grounded-SAM-2 repository, and change it to groundingdino.

# 6. For OSError: image file is truncated (18 bytes not processed)
# Add the following lines to any files that load images, like generate files and transform.py
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```


### CLIP Text Embeddings

Download the pre-generated CLIP text embeddings from the [Releases](https://github.com/hustvl/GaussTR/releases/) page.  Alternatively, you can generate custom embeddings by referring to [mmpretrain #1737](https://github.com/open-mmlab/mmpretrain/pull/1737) or [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO).

**Tip:** The default prompts have not been delicately tuned. Customizing them may yield improved results.

## Usage

|                               Model                               |  IoU  |  mIoU |                                                 Checkpoint                                                 |
| ----------------------------------------------------------------- | ----- | ----- | ---------------------------------------------------------------------------------------------------------- |
| [GaussTR-FeatUp](configs/gausstr_featup.py)                       | 45.19 | 11.70 | [checkpoint](https://github.com/hustvl/GaussTR/releases/download/v1.0/gausstr_featup_e24_miou11.70.pth)    |
| [GaussTR-Talk2DINO](configs/gausstr_talk2dino.py)<sup>*New*</sup> | 44.54 | 12.27 | [checkpoint](https://github.com/hustvl/GaussTR/releases/download/v1.0/gausstr_talk2dino_e20_miou12.27.pth) |

### Training

**Tip:** Due to the current lack of optimization for voxelization operations, evaluation during training can be time-consuming. To accelerate training, consider evaluating using the `mini_train` set or reducing the evaluation frequency.

```bash
PYTHONPATH=. mim train mmdet3d [CONFIG] [-l pytorch -G [GPU_NUM]]
```

```bash
# For Talk2DINO error with torch.hub.load
rm -rf /nethome/lzhao360/.cache/torch/hub/facebookresearch_dinov2_main
python tools/load_dinov2.py # For checking whether the model is loaded correctly
```

### Testing

```bash
PYTHONPATH=. mim test mmdet3d [CONFIG] -C [CKPT_PATH] [-l pytorch -G [GPU_NUM]]
```

### Visualization

To enable visualization, run the testing with the following included in the config:

```python
custom_hooks = [
    dict(type='DumpResultHook'),
]
```

After testing, visualize the saved `.pkl` files with:

```bash
python tools/visualize.py [PKL_PATH] [--save]
```

## Citation

If our paper and code contribute to your research, please consider starring this repository :star: and citing our work:

```BibTeX
@inproceedings{GaussTR,
    title     = {GaussTR: Foundation Model-Aligned Gaussian Transformer for Self-Supervised 3D Spatial Understanding},
    author    = {Haoyi Jiang and Liu Liu and Tianheng Cheng and Xinjie Wang and Tianwei Lin and Zhizhong Su and Wenyu Liu and Xinggang Wang},
    year      = 2025,
    booktitle = {CVPR}
}
```

## Acknowledgements

This project is built upon the pioneering work of [FeatUp](https://github.com/mhamilton723/FeatUp), [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO), [MaskCLIP](https://github.com/chongzhou96/MaskCLIP) and [gsplat](https://github.com/nerfstudio-project/gsplat). We extend our gratitude to these projects for their contributions to the community.

## License

Released under the [MIT](LICENSE) License.
