# Multimodal Range View Based Semantic Segmentation
Code for our project "Multimodal Range View Based Semantic Segmentation" for the course "Deep Learning for 3D Perception" at the Technical University of Munich under supervision of Prof. Angela Dai.

## Prepare:
Download SemanticKITTI from [official web](http://www.semantic-kitti.org/dataset.html). Download SemanticPOSS from [official web](http://www.poss.pku.edu.cn./download.html).

## Usage：
### Train：
- SemanticKITTI:

    `python train.py -d /your_dataset -ac config/arch/senet-512.yml -n senet-512`

    Note that the following training strategy is used due to GPU and time constraints, see [kitti.sh](https://github.com/huixiancheng/SENet/blob/main/kitti.sh) for details.

    First train the model with 64x512 inputs. Then load the pre-trained model to train the model with 64x1024 inputs, and finally load the pre-trained model to train the model with 64x2048 inputs.

### Infer and Eval：
- SemanticKITTI:

    `python infer.py -d /your_dataset -l /your_predictions_path -m trained_model -s valid/test`
    
    Eval for valid sequences:

    `python evaluate_iou.py -d /your_dataset -p /your_predictions_path`

    This will generate both predictions and mIoU results.

### Visualize Example:


- Visualize GT:

  `python visualize.py -w kitti -d /your_dataset -s what_sequences`

- Visualize Predictions:

  `python visualize.py -w kitti -d /your_dataset -p /your_predictions -s what_sequences`


## Pretrained Models and Logs:
Our pre-trained models can be found [here](https://drive.google.com/drive/folders/18lHtsK8KS-kRpsY5zd32y_7Ps0qVC5o9?usp=sharing).

## Acknowledgments：
Our codebase originates from [CENet](https://github.com/huixiancheng/CENet). For the fusion model we use code from [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion), while we follow the Hugging Face implementation of [Mask2Former](https://huggingface.co/docs/transformers/main/model_doc/mask2former) as RGB backbone. For initialization, we utilize the pre-trained Mask2Former models trained on the Cityscapes dataset for semantic segmentation.
