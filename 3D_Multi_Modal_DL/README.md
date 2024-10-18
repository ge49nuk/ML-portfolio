# 3D-refined SAM-6D

Tested with:
- Python=3.8.12
- pytorch 2.1.1
- CUDA 11.8

## Environment

To be able to fully use this repository please setup your environment as follows:

1. Install dependencies:
```shell
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

2. Download model weights of [Fast Segmenting Anything](https://github.com/CASIA-IVA-Lab/FastSAM):
```shell
python download_fastsam.py
```

3. Download model weights of ViT pre-trained by [DINOv2](https://github.com/facebookresearch/dinov2):
```shell
python download_dinov2.py
```

4. Compile C++ and CUDA scripts:

```shell
cd roitr_cpp_wrappers
cd pointops
python setup.py install
cd ..
cd ..
```

## Usage

### Train RoITr

To train RoITr on a BOP-dataset follow the instructions below:

1. Gain access to the PoseVerse dataset. The folder structure should include the following subfolders:
```
DATASET_NAME
├─ models
│  ├─ obj_OBJ_ID.ply
├─ train_pbr
│  ├─ SCENE_ID
│  │  ├─ scene_camera.json
│  │  ├─ scene_gt.json
│  │  ├─ scene_gt_info.json
│  │  ├─ depth
│  │  ├─ mask_visib
│  │  ├─ rgb
```

2. Adapt the root directory in configs/roitr/train/poseverse.yaml.

3. Start training by using:
```shell
python roitr_main.py configs/roitr/train/poseverse.yaml
```

### Test RoITr and refined SAM-6D:

1. Download a test dataset from the official [BOP-Website](https://bop.felk.cvut.cz/datasets/) and place it under ../Data/BOP/DATASET_NAME (you can change the root dir inside configs/user/default.yaml). The folder structure should include the following subfolders:
```
DATASET_NAME
├─ models
│  ├─ obj_OBJ_ID.ply
├─ test / test_primesense
│  ├─ SCENE_ID
│  │  ├─ scene_camera.json
│  │  ├─ scene_gt.json
│  │  ├─ scene_gt_info.json
│  │  ├─ depth
│  │  ├─ mask_visib
│  │  ├─ rgb
```

2. Start the test:
**RoITr**
```shell
python roitr_main.py configs/roitr/test/DATASET_NAME.yaml
```

**SAM-6D**

a. Set dataset_name inside configs/run_inference.yaml to DATASET_NAME.

b. Set the path to the RoITr checkpoint inside configs/roitr/inference.yaml to the .pth file inside the roitr_snapshot folder. (RoITr needs to be trained first)

c. Execute the inference script:
```shell
python sam6d_inference.py
```

## More info

For more information on the project, please refer to https://confluence.mvtec.com/display/RS/%28MA%29+Multi+modal+Deep+Learning+models+for+combined+image+and+3D+data+analysis.

