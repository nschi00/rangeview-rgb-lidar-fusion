# Multimodal Range View Based Semantic Segmentation
Code for our project "Multimodal Range View Based Semantic Segmentation" of the course "Deep Learning for 3D Perception" at the Technical University of Munich under supervision of Prof. Angela Dai.

## Prepare:
Download SemanticKITTI from their [official website](http://www.semantic-kitti.org/dataset.html).

## Usage：
### Train：
- Lidar backbone with Range Augmentations (RA):
    - 512 x 64 range-view (RV) resolution:
      ```bash
      python train.py -d /path/to/SemanticKITTI/dataset -ac config/arch/cenet_512.yml \
          -n cenet_512_RA
      ```
    - 1024 x 64 RV resolution (retrain from 512 x 64 checkpoint as the authors of CENet recommend):
        ```bash
        python train.py -d /path/to/SemanticKITTI/dataset -ac config/arch/cenet_1024.yml \
            -p /path/to/cenet_512_RA -n cenet_1024_RA
        ```
        
- RGB backbone fine-tuning on SemanticKITTI dataset with range-view labels:
    - for usage with 512 x 64 model:
      ```bash
      python train.py -d /path/to/SemanticKITTI/dataset -ac config/arch/mask2former_512.yml \
          -n mask2former_512
      ```
    - for usage with 1024 x 64 model:
      ```bash
      python train.py -d /path/to/SemanticKITTI/dataset -ac config/arch/mask2former_1024.yml \
          -n mask2former_1024
      ```

- Fusion Model:
    - 512 x 64 range-view (RV) resolution:
      ```bash
      python train.py -d /path/to/SemanticKITTI/dataset -ac config/arch/fusion_512.yml \
          -n fusion_512
      ```
    - 1024 x 64 RV resolution:
      ```bash
      python train.py -d /path/to/SemanticKITTI/dataset -ac config/arch/fusion_1024.yml \
          -n fusion_1024
      ```

### Infer and Evaluation：
- Infer:
  ```bash
  python infer.py -d /path/to/SemanticKITTI/dataset -l /path/to/save/predictions/in \
      -m path/to/trained_model
  ```

- Evalulation:
    - Lidar and fusion models:
      ```bash
      python evaluate_iou.py -d /path/to/SemanticKITTI/dataset -p /path/to/predictions
      ```
    - RGB models:
      ```bash
      python evaluate_iou_rgb.py -d /path/to/SemanticKITTI/dataset -p /path/to/predictions
      ```

### Visualize Example:
- Visualize GT:
  ```bash
  python visualize.py -w kitti -d /path/to/SemanticKITTI/dataset -s which_sequences
  ```

- Visualize Predictions:
  ```bash
  python visualize.py -w kitti -d /path/to/SemanticKITTI/dataset -p /path/to/predictions \
  -s which_sequences
  ```


## Pretrained Models and Logs:
Our pre-trained models can be found [here](https://drive.google.com/drive/folders/18lHtsK8KS-kRpsY5zd32y_7Ps0qVC5o9?usp=sharing).

## Acknowledgments：
Our codebase originates from [CENet](https://github.com/huixiancheng/CENet). For the fusion model we use code from [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion), while we follow the Hugging Face implementation of [Mask2Former](https://huggingface.co/docs/transformers/main/model_doc/mask2former) as RGB backbone. For initialization, we utilize the pre-trained Mask2Former models trained on the Cityscapes dataset for semantic segmentation.
