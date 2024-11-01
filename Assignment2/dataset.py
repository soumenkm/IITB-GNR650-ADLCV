import torch, pickle, tqdm, os, pickle
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Union

class AwA2Dataset(Dataset):
    def __init__(self, visual_feature_path: Path, num_train: int, is_train: bool):
        super(AwA2Dataset, self).__init__()
        self.path = visual_feature_path
        self.data = pickle.load(open(self.path, "rb"))
        self.data = {k: self.data[k] for k in sorted(self.data.keys())}
        self.classes = sorted(list(set([i.split("/")[0] for i in self.data.keys()])))
        self.labels = range(len(self.classes))
        self.class_to_label = dict(zip(self.classes, self.labels))
        self.label_to_class = dict(zip(self.labels, self.classes))
        self.train_classes = self.classes[0:num_train]
        self.test_classes = self.classes[num_train:]
        
        if is_train:
            self.ds = {k: v for k, v in self.data.items() if k.split("/")[0] in self.train_classes}
        else:
            self.ds = {k: v for k, v in self.data.items() if k.split("/")[0] in self.test_classes}
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
        item = list(self.ds.values())[index]
        feature = item["feature"]
        label = torch.tensor(self.class_to_label[item["label"]])
        return (feature.float(), label.item())
    