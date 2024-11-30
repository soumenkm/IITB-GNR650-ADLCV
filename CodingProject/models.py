import os, torch, random
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
from transformers import AutoModel
import torchinfo
torch.manual_seed(42)
from typing import List, Tuple, Union

class LoRALayer(torch.nn.Module):
    def __init__(self, rank: int, alpha: float, d_in: int, d_out: int):  
        super(LoRALayer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.alpha = alpha
        self.rank = rank
        
        self.A = torch.nn.Parameter(
            data=torch.normal(mean=0, std=0.01, size=(self.d_in, self.rank)), 
            requires_grad=True
        )
        self.B = torch.nn.Parameter(
            data=torch.zeros(size=(self.rank, self.d_out)),
            requires_grad=True
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        delta_W = torch.matmul(self.A, self.B) * (self.alpha / self.rank)
        z = torch.matmul(x, delta_W)
        return z
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float):
        super(LinearWithLoRA, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.linear = linear
        self.d_in = self.linear.in_features
        self.d_out = self.linear.out_features
        self.lora = LoRALayer(rank=self.rank, alpha=self.alpha, d_in=self.d_in, d_out=self.d_out)

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        z1 = self.linear(x) 
        z2 = self.lora(x) 
        z = z1 + z2
        return z

class VisionModelForCLS(nn.Module):
    def __init__(self, device: torch.device, model_name: str, num_classes: int):
        super(VisionModelForCLS, self).__init__()
        self.device = device
        self.model_name = model_name
        self.base = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.classifier = nn.Linear(self.base.config.hidden_size, num_classes).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def get_layers(self) -> torch.nn.ModuleList:
        if "dino" in self.model_name:
            layers = self.base.encoder.layer
        else:
            raise NotImplementedError("Invalid model name!")
        return layers
    
    def _unfreeze_layers(self, last_num_layers: int) -> None:
        for val in self.base.parameters():
            val.requires_grad_(False)
        for val in self.classifier.parameters():
            val.requires_grad_(True)
        all_layers = self.get_layers()        
        for layer in all_layers[-last_num_layers:]:
            for val in layer.parameters():
                val.requires_grad_(True)
            
    def calc_num_params(self) -> None:
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"Train efficiency: {train_params * 100 / total_params:.3f}%")
    
    def forward(self, pixels: torch.tensor, labels):
        """pixels.shape = (b, 3, 224, 224)"""
        outputs = self.base(pixel_values=pixels) 
        hidden_state = outputs.last_hidden_state[:, 0, :]  # CLS token output
        logits = self.classifier(hidden_state) # (b, c)
        if labels is not None:
            labels = labels.to(self.device)  
            loss = self.loss_fn(logits, labels) 
        else:
            loss = None
        return {"logits": logits, "loss": loss}

class VisionModelForCLSWithLoRA(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, num_classes: int, lora_rank: int, lora_alpha: float, linear_names: List[str]):
        """linear_names = Any[query, key, value, dense, fc1, fc2]"""
        super(VisionModelForCLSWithLoRA, self).__init__()
        self.device = device
        self.model_name = model_name
        self.num_class = num_classes
        self.linear_names = linear_names
        self.model = VisionModelForCLS(device=self.device, model_name=self.model_name, num_classes=self.num_class)
        
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        self.rank = lora_rank
        self.alpha = lora_alpha
        self.apply_lora(rank=self.rank, alpha=self.alpha)
    
    def apply_lora(self, rank: int, alpha: float) -> None:        
        VisionModelForCLSWithLoRA.replace_linear_with_lora(device=self.device, model=self.model.base, rank=rank, alpha=alpha, linear_names=self.linear_names)            

    @staticmethod
    def replace_linear_with_lora(device: torch.device, model: torch.nn.Module, rank: int, alpha: float, linear_names: List[str]):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                if any(proj in name for proj in linear_names):
                    linear_lora = LinearWithLoRA(module, rank, alpha)
                    setattr(model, name, linear_lora) # parent is model, child is module
            else:
                VisionModelForCLSWithLoRA.replace_linear_with_lora(device, module, rank, alpha, linear_names)
     
    def calc_num_params(self) -> None:
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"LoRA efficiency: {train_params * 100 / total_params:.3f}%")
    
    def forward(self, pixels: torch.tensor, labels: Union[torch.tensor, None]) -> torch.tensor:
        assert list(pixels.shape).__len__() == 4, "inputs rank must be 4 and inputs.shape = (b, C, W, H)"
        prediction_output = self.model(pixels, labels) # (b, c)
        return prediction_output

if __name__ == "__main__":
    model_name = "facebook/dinov2-base"
    model = VisionModelForCLSWithLoRA("cuda", model_name, 51, 8, 16, ["fc1", "fc2"]).to("cuda")
    a = torch.rand(size=(4, 3, 224, 224)).to("cuda")
    print(model)
    b = model(a, labels=None)
    print(b)
    print("DONE")