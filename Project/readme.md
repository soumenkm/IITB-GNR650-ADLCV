# Vision Model Fine-Tuning with MillionAID Dataset

GitHub Link: [GitHub](https://github.com/soumenkm/IITB-GNR650-ADLCV/tree/main/Project)
This project fine-tunes a Vision Transformer model (e.g., DINOv2) with LoRA (Low-Rank Adaptation) and layer unfreezing techniques on the MillionAID dataset. The model classifies images across 51 classes, and the training process is facilitated using the Hugging Face Transformers library.

## Table of Contents
- [Requirements](#requirements)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
- [Evaluating the Model](#evaluating-the-model)
- [Using other models](#using-other-models)

### Requirements
Make sure to install the necessary libraries:

```bash
pip install torch torchvision transformers datasets evaluate wandb tqdm pillow
```

### Setup
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/soumenkm/IITB-GNR650-ADLCV.git
    cd IITB-GNR650-ADLCV
    ```

2. **Setup W&B**:
    Log in to your [Weights & Biases](https://wandb.ai/) account and configure logging:
    ```bash
    wandb login
    ```

### Data Preparation
The script uses the MillionAID dataset, which is directly loaded using `load_dataset` from Hugging Face. No additional data preprocessing is required, as the dataset will be automatically downloaded and processed when running the code.

### Configuration
Modify parameters in the `config` dictionary in `main()` function of `train.py` to customize training:
- `model_name`: The model architecture to use (currently only supports "facebook/dinov2-base").
- `finetune_type`: Specify "lora" or "layer" based on your preference.
  - For `lora`, adjust `lora_rank`, `lora_alpha`, and `lora_linear_names`.
  - For `layer`, set `last_num_layers`.
- `batch_size`, `num_epochs`, `initial_lr`, and `weight_decay` are also configurable.

### Running the Code
Use the following command to train the model:

```bash
CUDA_VISIBLE_DEVICES=0 python Project/train.py \
    --model_name "facebook/dinov2-base" \
    --finetune_type "lora" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_linear_names "query,key,value,dense" \
    --num_epochs 20 \
    --batch_size 16 \
    --frac 1.0 \
    --num_classes 51 \
    --initial_lr 1e-5 \
    --max_grad_norm 10.0 \
    --weight_decay 0.1 \
    --adam_beta1 0.95 \
    --adam_beta2 0.999 \
    --grad_acc_steps 1 \
    --num_ckpt_per_epoch 1 \
    --wandb_log True
```

### Evaluating the Model
To evaluate the model on the test dataset, you can use the provided `evaluate_model` function in `test.py`, which loads a pre-trained model from a specified checkpoint and evaluates its performance.

#### Usage

1. **Configure Evaluation Parameters**:
   Ensure you have a configuration file (`master_config.pkl`) and a saved checkpoint (e.g., `checkpoint-100/pytorch_model.bin`) from your training process.

2. **Run Evaluation**:
   Use the following command to test the model:

   ```bash
    CUDA_VISIBLE_DEVICES=0 python Project/test.py \
    --config_path "Project/outputs/ckpt/dinov2-base_finetune/lora_0.10_1.0e-05_r8/master_config.pkl" \
    --checkpoint_name "checkpoint-200/pytorch_model.bin"
    ```

### Using Other Models

The current implementation is designed for the **DINOv2** model architecture from Facebook. However, you can easily adapt this code to work with other vision models, such as **ViT**, **ResNet**, or other models available in the Hugging Face library.

#### Steps to Incorporate Other Models

1. **Choose a Model Compatible with Hugging Face Transformers**:  
   Select a model available in the [Hugging Face Model Hub](https://huggingface.co/models) that is suitable for image classification and compatible with the `AutoModel` class, such as `google/vit-base-patch16-224` for Vision Transformer (ViT) or other models like Swin Transformer.

2. **Modify the `VisionModelForCLS` Class**:  
   Ensure that the new model has a compatible architecture for image processing (i.e., it should have `last_hidden_state` or a similar final representation layer for classification). Otherwise you may need to modify the `forward` method of `VisionModelForCLS` class.

3. **Adjust Layer Freezing Logic** (if needed):  
   If your new model has a different structure for layers (e.g., `encoder.layers` instead of `encoder.layer`), modify the `get_layers` method in the `VisionModelForCLS` class to correctly retrieve the layers for layer freezing or fine-tuning.

   ```python
   def get_layers(self) -> torch.nn.ModuleList:
       if "vit" in self.model_name:
           layers = self.base.encoder.layer
       elif "resnet" in self.model_name:
           layers = self.base.layer
       else:
           raise NotImplementedError("Invalid model name or structure!")
       return layers
   ```

4. **Apply LoRA for New Linear Layers**:  
   For LoRA integration, ensure that the linear layers to which LoRA is applied exist in the new model. You may need to update the `linear_names` parameter with specific layer names that match the new model’s structure (e.g., `query`, `key`, `value` for transformers or `fc` for ResNet).

