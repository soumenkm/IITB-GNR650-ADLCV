DATASET = 'AWA2' # One of ["AWA1", "AWA2", "APY", "CUB", "SUN"]
USE_CLASS_STANDARTIZATION = True # i.e. equation (9) from the paper
USE_PROPER_INIT = True # i.e. equation (10) from the paper

import argparse
import numpy as np; np.random.seed(1)
from collections import defaultdict
import torch; torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
from time import time
from tqdm import tqdm
from scipy import io
from torch.utils.data import DataLoader
from dataset import AwA2Dataset
from pathlib import Path
import pickle
from numpy.linalg import norm
import torchinfo

class ClassStandardization(nn.Module):
    """
    Class Standardization procedure from the paper.
    Conceptually, it is equivalent to nn.BatchNorm1d with affine=False,
    but for some reason nn.BatchNorm1d performs slightly worse.
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        
        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad=False)
    
    def forward(self, class_feats):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_feats.mean(dim=0)
            batch_var = class_feats.var(dim=0)
            
            # Normalizing the batch
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)
            
            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            # Using accumulated statistics
            # Attention! For the test inference, we cant use batch-wise statistics,
            # only the accumulated ones. Otherwise, it will be quite transductive
            result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)
        
        return result

class CNZSLModel(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(attr_dim, hid_dim),
            nn.ReLU(),
            
            nn.Linear(hid_dim, hid_dim),
            ClassStandardization(hid_dim) if USE_CLASS_STANDARTIZATION else nn.Identity(),
            nn.ReLU(),
            
            ClassStandardization(hid_dim) if USE_CLASS_STANDARTIZATION else nn.Identity(),
            nn.Linear(hid_dim, proto_dim),
            nn.ReLU(),
        )
        
        if USE_PROPER_INIT:
            weight_var = 1 / (hid_dim * proto_dim)
            b = np.sqrt(3 * weight_var)
            self.model[-2].weight.data.uniform_(-b, b)
        
    def forward(self, x, attrs):
        protos = self.model(attrs)
        x_ns = 5 * x / x.norm(dim=1, keepdim=True) # [batch_size, x_dim]
        protos_ns = 5 * protos / protos.norm(dim=1, keepdim=True) # [num_classes, x_dim]
        logits = x_ns @ protos_ns.t() # [batch_size, num_classes]
        
        return logits
    

def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-Shot Learning Model Configuration")
    parser.add_argument('--visual_encoder', type=str, required=True, help="Specify the visual encoder model (e.g., 'vit' or 'resnet50')")
    parser.add_argument('--text_encoder', type=str, required=True, help="Specify the text encoder model (e.g., 'fasttext' or 'word2vec')")
    args = parser.parse_args()
    visual_model, text_model = args.visual_encoder, args.text_encoder
    
    if "vit" in visual_model:
        visual_model = "vit-base-patch16-224-in21k"
    elif "resnet" in visual_model:
        visual_model = "resnet-50"
    else:
        raise ValueError("Invalid visual model!")
    
    if "word2vec" in text_model:
        text_model = "word2vec-google-news-300"
    elif "fasttext" in text_model:
        text_model = "fasttext-wiki-news-subwords-300"
    else:
        raise ValueError("Invalid text model!")
    
    cwd = str(Path.cwd())
    num_train = 25
    DEVICE = "cuda"
    ds_train = AwA2Dataset(Path(f"{cwd}/Assignment2/data/visual_features/{visual_model}.pkl"),
                        is_train=True,
                        num_train=num_train)
    ds_test = AwA2Dataset(Path(f"{cwd}/Assignment2/data/visual_features/{visual_model}.pkl"),
                        is_train=False,
                        num_train=num_train)

    attrs_dict = pickle.load(open(Path(f"{cwd}/Assignment2/data/text_features/{text_model}.pkl"), "rb"))
    attrs = [None for i in range(50)]
    for k, v in attrs_dict.items():
        if k in ds_train.classes:
            attrs[ds_train.class_to_label[k]] = v/norm(v)

    attrs = torch.tensor(attrs).to(DEVICE).float()

    train_dataloader = DataLoader(ds_train, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=2048)


    print(f'\n<=============== Starting training ===============>')

    start_time = time()
    model = CNZSLModel(attrs.shape[1], 1024, ds_train[0][0].shape[0]).to(DEVICE)
    optim = torch.optim.Adam(model.model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, gamma=0.1, step_size=25)
    torchinfo.summary(model)

    train_step = 0
    for epoch in tqdm(range(5)):
        model.train()
        
        for i, batch in enumerate(train_dataloader):
            feats = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            logits = model(feats, attrs[:num_train])
            loss = F.cross_entropy(logits, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_step += 1
        
        scheduler.step()

    print(f'Training is done! Took time: {(time() - start_time): .1f} seconds')

    model.eval() # Important! Otherwise we would use unseen batch statistics
    logits = [model(x.to(DEVICE), attrs).cpu() for x, _ in test_dataloader]
    logits = torch.cat(logits, dim=0)

    preds_zsl_u = logits[:, num_train:].argmax(dim=1).numpy() + num_train
    test_labels = [ds_test[i][1] for i in range(len(ds_test))]

    # Group predictions by class
    class_correct_counts = defaultdict(int)
    class_total_counts = defaultdict(int)

    for i, true_label in enumerate(test_labels):
        if true_label in range(num_train, 50):  # Check if label is in unseen classes
            class_total_counts[true_label] += 1
            if preds_zsl_u[i] == true_label:
                class_correct_counts[true_label] += 1

    # Calculate mean accuracy over unseen classes
    class_accuracies = [
        class_correct_counts[c] / class_total_counts[c]
        for c in class_total_counts if class_total_counts[c] > 0
    ]
    zsl_unseen_acc = np.mean(class_accuracies)

    print(f'ZSL-U: {zsl_unseen_acc * 100:.02f}')

if __name__ == "__main__":
    main()