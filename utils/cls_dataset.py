"""
    Implementation of several classification dataset loader.
"""

import io
import os
from typing import Any, Callable, Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pydicom
from sklearn.preprocessing import LabelBinarizer

try:
    from petrel_client.client import Client
except:
    print("petrel_client is not installed, petrel oss cannot be used currently")

class ChestXRay(Dataset):
    def __init__(self, 
        data_path:str="/path/to/chestxray", 
        train:bool=True,
        transform=None,
        petrel_oss:bool=False,
        petrel_url:str="/petrel/path/to/chestxray",
        petrel_conf:str="~/petreloss.conf",
        onehot:bool=True
    ):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.petrel_oss = petrel_oss
        self.petrel_url = petrel_url
        self.petrel_conf = petrel_conf
        self.onehot = onehot
        self._load_data()

    def _load_data(self):
        if self.petrel_oss and self.petrel_url is None:
            raise ValueError("Not specified petrel URL")
        self.classes = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
            "No_Finding"
        ]
        if not self.petrel_oss:
            self.img_folder = os.path.join(self.data_path, "chestxray14_512")
            labels_path = os.path.join(self.data_path, "chestxray14_train_val.csv" if self.train else "chestxray14_test.csv")
            labels_dict = pd.read_csv(labels_path).to_dict()
            self.data_ids = labels_dict["id"]
            labels = []
            for c in self.classes:
                labels.append(torch.tensor([labels_dict[c][_k] for _k in sorted(labels_dict[c].keys())], dtype=torch.long))
            labels_onehot = torch.stack(labels, axis=1)
            # self.labels = torch.argmax(labels_onehot, dim=1)
        else:
            self.img_folder = os.path.join(self.petrel_url, "images")
            labels_path = os.path.join(self.petrel_url, "chestxray14_train_val.csv" if self.train else "chestxray14_test.csv")
            self.client = Client(self.petrel_conf)
            with io.BytesIO(self.client.get(labels_path)) as fp:
                labels_dict = pd.read_csv(fp).to_dict()
                # label_file = csv.reader(fp)
                # lines = [line for line in label_file]
                # labels_dict = pd.DataFrame(lines[1:], columns=lines[0]).to_dict()
            self.data_ids = labels_dict["id"]
            labels = []
            for c in self.classes:
                labels.append(torch.tensor([labels_dict[c][_k] for _k in sorted(labels_dict[c].keys())], dtype=torch.long))
            labels_onehot:torch.Tensor = torch.stack(labels, axis=1)
        self.labels = labels_onehot.to(torch.float32) if self.onehot else torch.argmax(labels_onehot, dim=1)
            

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        img_path = os.path.join(self.img_folder, self.data_ids[idx])
        if not self.petrel_oss:
            img = Image.open(img_path)
        else:
            img_val = self.client.get(img_path)
            img_arr = np.fromstring(img_val, np.uint8)
            img = Image.fromarray(cv2.cvtColor(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self) -> int:
        return len(self.labels)



class MNIST_REP(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__()
        self.mnist = datasets.MNIST(root, train, transform, target_transform, download)
        self.classes = self.mnist.classes
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        img, target = self.mnist.__getitem__(index)
        return torch.tile(img, [3, 1, 1]), target
    
    def __len__(self) -> int:
        return len(self.mnist)


class ImageNet(Dataset):
    def __init__(self, root: str = "/path/to/imagenet", split: str = "train", transform: Optional[Callable] = None) -> None:
        super().__init__()
        if split not in ["train", "val"]:
            raise NotImplementedError(f"{split}: is not supported or the labels are hidden")
        root = os.path.join(root, "images")
        # load data
        root = self.root = os.path.expanduser(root)
        self.split_root = os.path.join(root, split)
        self.transform = transform
        with open(os.path.join(root, "meta", split + ".txt"), "r") as fp:
            wnid_to_classes_lines = fp.readlines()
            wnid_to_classes = [line.strip().split(" ") for line in wnid_to_classes_lines]
            self.wnids = [line[0] for line in wnid_to_classes]
            self.classes = [int(line[1]) for line in wnid_to_classes]

    def __len__(self) -> int:
        return len(self.classes)
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        img = Image.open(os.path.join(self.split_root, self.wnids[index])).convert('RGB')
        label = self.classes[index]
        if self.transform:
            img = self.transform(img)
        return img, label


class MITIndoor(Dataset):
    def __init__(self, root: str = "/path/to/indoor", split: str = "train", transform: Optional[Callable] = None) -> None:
        super().__init__()
        if split not in ["train", "val"]:
            raise NotImplementedError(f"{split}: is not supported or the labels are hidden")
        self.images_root = os.path.join(root, "Images")
        self.transform = transform
        with open(os.path.join(root, "TrainImages.txt" if split == "train" else "TestImages.txt"), "r") as fp:
            lines = fp.readlines()
            self.imgs_path = [line.strip() for line in lines]
            self.labels_str = [line.strip().split("/")[0] for line in lines]
            self.classes = np.unique(self.labels_str)
            str_to_cls_idx = {c: i for i, c in enumerate(self.classes)}
            self.labels = [str_to_cls_idx[label_str] for label_str in self.labels_str]
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        img = Image.open(os.path.join(self.images_root, self.imgs_path[index])).convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label


class ISIC2020(Dataset):
    def __init__(self, root:str="/path/to/isic2020", transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.classes = ["benign", "malignant"]
        metadata_path = os.path.join(root, "ISIC_2020_Training_GroundTruth_v2.csv")
        self.df = pd.read_csv(metadata_path)
        lb = LabelBinarizer()
        self.df["enc_label"] = lb.fit_transform(self.df["benign_malignant"]).tolist()
        self.transform = transform
        self.data_folder = os.path.join(root, "train")
        
        def fix_enc(el):
            if el[0] == 0:
                return [0, 1]
            
            if el[0] == 1:
                return [1, 0]
            
            raise ValueError("Value not 0 or 1")
        
        self.df["enc_label"] = self.df["enc_label"].apply(fix_enc)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        image = pydicom.dcmread(os.path.join(self.data_folder, self.df.iloc[index]["image_name"] + ".dcm")).pixel_array
        if self.transform:
            image = self.transform(image)
        enc_label = torch.tensor(self.df.iloc[index]["enc_label"]).long()
        return image.float(), enc_label

class ISIC2019(Dataset):
    def __init__(self, root:str="/path/to/isic2019", transform: Optional[Callable] = None) -> None:
        super().__init__()
        metadata_path = os.path.join(root, "ISIC_2019_Training_GroundTruth.csv")
        self.df = pd.read_csv(metadata_path)
        self.classes = list(self.df.columns.values)[1:]
        str_to_cls_idx = {c: i for i, c in enumerate(self.classes)}
        for label in self.classes:
            self.df.loc[self.df[label] == 1.0, 'label_str'] = label
            self.df.loc[self.df[label] == 1.0, 'label'] = str_to_cls_idx[label]
        self.transform = transform
        self.data_folder = os.path.join(root, "ISIC_2019_Training_Input")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        image = Image.open(os.path.join(self.data_folder, self.df.iloc[index]["image"] + ".jpg")).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.df.iloc[index]["label"]).long()
        return image.float(), label


class KvasirCapsule(Dataset):
    def __init__(self, root:str="/path/to/kvasir_capsule", transform:Optional[Callable]=None) -> None:
        super().__init__()
        self.root = root
        self.classes = os.listdir(root)
        self.cls_str_to_idx = {category:idx for idx, category in enumerate(self.classes)}
        self.transform = transform
        self.img_paths:List[str] = []
        self.labels:List[int] = []
        for category in self.classes:
            label = self.cls_str_to_idx[category]
            for img_path in os.listdir(os.path.join(root, category)):
                self.img_paths.append(os.path.join(category, img_path))
                self.labels.append(label)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        image = Image.open(os.path.join(self.root, self.img_paths[index])).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label
        

class CRC100K(Dataset):
    def __init__(self, root:str="/path/to/NCT-CRC-HE-100K", transform:Optional[Callable]=None) -> None:
        super().__init__()
        self.root = root
        self.classes = os.listdir(root)
        cls_str_to_idx = {category:idx for idx, category in enumerate(self.classes)}
        self.transform = transform
        self.img_paths:List[str] = []
        self.labels:List[int] = []
        for category in self.classes:
            label = cls_str_to_idx[category]
            for img_path in os.listdir(os.path.join(root, category)):
                self.img_paths.append(os.path.join(category, img_path))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index) -> Tuple[Any, Any]:
        image = Image.open(os.path.join(self.root, self.img_paths[index])).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label
        



def test_chestray():
    dataset = ChestXRay(train=False, transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False)
    for batch in dataloader:
        print(len(dataloader))
        print(batch[0].shape, batch[1].shape)
        print(batch[0][0])
        exit()


def test_mnist_rep():
    dataset = MNIST_REP(os.path.join("/path/to/data", "mnist"), download=False, train=True, transform=transforms.ToTensor())
    print(dataset[0][0].shape)
    dataset = datasets.CIFAR10(os.path.join("/path/to/data", "cifar10"), download=False, train=True, transform=transforms.ToTensor())
    print(dataset[0][0].shape)

def test_indoor():
    train_transform = transforms.Compose([
            transforms.RandomChoice([transforms.Resize(256), transforms.Resize(480)]),
            transforms.RandomCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset = MITIndoor(transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False)
    for batch in dataloader:
        x, y = batch
        print(y)
        exit()


def test_kvasir_capsule():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = KvasirCapsule(transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False)
    for batch in dataloader:
        x, y = batch
        print(x.shape)
        print(y)
        exit()


def test_crc_100k():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CRC100K(transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    for batch in dataloader:
        x, y = batch
        print(x.shape)
        print(y)
        exit()


def test_isic_2020():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ISIC2020(transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    for batch in dataloader:
        x, y = batch
        print(x.shape)
        print(y)
        exit()


if __name__ == "__main__":
    test_isic_2020()