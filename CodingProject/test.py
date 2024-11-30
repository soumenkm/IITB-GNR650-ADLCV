import torch, pickle, os, argparse
from transformers import Trainer, DefaultDataCollator, TrainingArguments
import numpy as np
from pathlib import Path
from dataset import MillionAIDataset
from models import VisionModelForCLSWithLoRA, VisionModelForCLS
import evaluate
from train import FineTuner

def evaluate_model(device: torch.device, config_path: Path, checkpoint_name: str):
    with open(config_path, 'rb') as f:
        config_data = pickle.load(f)
    config = config_data["config"]
    
    # Initialize model based on the finetune type
    if config["finetune_type"] == "lora":     
        model = VisionModelForCLSWithLoRA(
            device=device,
            model_name=config["model_name"],
            num_classes=config["num_classes"],
            lora_rank=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            linear_names=config["lora_linear_names"]
        )
    elif config["finetune_type"] == "layer":
        model = VisionModelForCLS(
            device=device,
            model_name=config["model_name"],
            num_classes=config["num_classes"]
        )
    else:
        raise ValueError("Invalid finetune type!")
    
    # Load model checkpoint
    checkpoint_path = Path(config_path.parent, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    
    # Prepare evaluation dataset
    eval_ds = MillionAIDataset(frac=config["frac"], is_train=False)
    data_collator = DefaultDataCollator(return_tensors="pt")
    eval_args = TrainingArguments(
        output_dir="eval_res",
        per_device_eval_batch_size=config["batch_size"],
        report_to="none"  # Disable W&B reporting during evaluation
    )
    
    # Define Trainer for evaluation
    trainer = Trainer(
        args=eval_args,
        model=model,
        data_collator=data_collator,
        eval_dataset=eval_ds,
        compute_metrics=FineTuner.compute_metrics  # Use the compute_metrics defined in FineTuner
    )
    
    # Run evaluation
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    return eval_results

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with user-specified configuration and checkpoint.")

    # Arguments for the device and paths
    parser.add_argument("--config_path", type=str, default="Project/outputs/ckpt/dinov2-base_finetune/lora_0.10_1.0e-05_r8/master_config.pkl", help="Path to the model configuration file.")
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint-200/pytorch_model.bin", help="Name of the checkpoint file.")

    args = parser.parse_args()
    print(args)
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    config_path = Path(Path.cwd(), args.config_path)
    checkpoint_name = args.checkpoint_name
    
    evaluate_results = evaluate_model(device=device, config_path=config_path, checkpoint_name=checkpoint_name)
    print("Test Accuracy:", evaluate_results['eval_accuracy'])
