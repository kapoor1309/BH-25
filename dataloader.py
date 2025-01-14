import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Dataset classes
class CustomCholecT50:
    def __init__(self, dataset_dir, train_videos, test_videos, normalize=True, split_ratio=0.8):
        self.dataset_dir = dataset_dir
        self.train_videos = train_videos
        self.test_videos = test_videos
        self.normalize = normalize
        self.split_ratio = split_ratio

        train_transform, test_transform = self.transform()
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.build_train_valid_datasets(self.train_transform)
        self.build_test_dataset(self.test_transform)

    def transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        op_test = [transforms.Resize((224, 224)), transforms.ToTensor()]
        op_train = [transforms.Resize((224, 224)), transforms.ToTensor()]

        if self.normalize:
            op_test.append(normalize)
            op_train.append(normalize)

        test_transform = transforms.Compose(op_test)
        train_transform = transforms.Compose(op_train)

        return train_transform, test_transform

    def build_train_valid_datasets(self, transform):
        iterable_dataset = []
        for video in self.train_videos:
            dataset = T50(
                img_dir=os.path.join(self.dataset_dir, f"videos/VID{str(video).zfill(2)}"),
                annotation_file=os.path.join(self.dataset_dir, f"labels/VID{str(video).zfill(2)}.json"),
                transform=transform
            )
            iterable_dataset.append(dataset)
        full_dataset = torch.utils.data.ConcatDataset(iterable_dataset)

        train_size = int(self.split_ratio * len(full_dataset))
        valid_size = len(full_dataset) - train_size
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

    def build_test_dataset(self, transform):
        iterable_dataset = []
        for video in self.test_videos:
            dataset = T50(
                img_dir=os.path.join(self.dataset_dir, f"videos/VID{str(video).zfill(2)}"),
                annotation_file=os.path.join(self.dataset_dir, f"labels/VID{str(video).zfill(2)}.json"),
                transform=transform
            )
            iterable_dataset.append(dataset)
        self.test_dataset = torch.utils.data.ConcatDataset(iterable_dataset)

    def build(self):
        return self.train_dataset, self.valid_dataset, self.test_dataset


class T50(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        with open(annotation_file, 'r') as file:
            annotations = json.load(file)
        self.annotations = annotations["annotations"]
        self.img_dir = img_dir
        self.transform = transform
        self.frame_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):
        frame_id = self.frame_ids[index]
        triplets = self.annotations[frame_id]

        basename = f"{str(frame_id).zfill(6)}.png"
        img_path = os.path.join(self.img_dir, basename)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img_path}")

        if self.transform:
            image = self.transform(image)

        triplet_label, tool_label, verb_label, target_label = self.get_binary_labels(triplets)
        return image, triplet_label, tool_label, verb_label, target_label

    def get_binary_labels(self, labels):
        tool_label = np.zeros([6])
        verb_label = np.zeros([10])
        target_label = np.zeros([15])
        triplet_label = np.zeros([100])

        for label in labels:
            triplet = label[0:1]
            if triplet[0] != -1.0:
                triplet_label[triplet[0]] = 1
            tool = label[1:7]
            if tool[0] != -1.0:
                tool_label[tool[0]] = 1
            verb = label[7:8]
            if verb[0] != -1.0:
                verb_label[verb[0]] = 1
            target = label[8:14]
            if target[0] != -1.0:
                target_label[target[0]] = 1

        return triplet_label, tool_label, verb_label, target_label
