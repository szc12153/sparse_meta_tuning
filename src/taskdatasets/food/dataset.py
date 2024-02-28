from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
from torchvision.datasets import Food101
import numpy as np
from PIL import Image
import os
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset


class FoodDataset(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = FoodClassDataset(root, meta_train=meta_train,
                                   meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                   transform=transform, class_augmentations=class_augmentations,
                                   download=download)
        super(FoodDataset, self).__init__(dataset, num_classes_per_task,
                                          target_transform=target_transform, dataset_transform=dataset_transform)

    @staticmethod
    def get_transform(split, input_size, aug=False):
        normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        resize_size = int(input_size * 256 / 224)
        if split == "train" and aug:
            return Compose([
                RandomResizedCrop(input_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ])
        else:
            return Compose([
                Resize(resize_size),
                CenterCrop(input_size),
                ToTensor(),
                normalize,
            ])


class FoodClassDataset(ClassDataset):
    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(FoodClassDataset, self).__init__(meta_train=meta_train,
                                               meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                               class_augmentations=class_augmentations)
        self.root = root
        self.transform = transform
        temp_dataaset = Food101(
            root=self.root,
            split=meta_split,
            download=False
        )
        self.labels = np.unique(temp_dataaset._labels).tolist()
        assert len(self.labels) == 101
        self.data = []
        for c in self.labels:
            self.data.append([x for x,y in zip(temp_dataaset._image_files,temp_dataaset._labels) if y == c])

        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return Dummy(index, data, label,
                     transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes


class Dummy(Dataset):
    def __init__(self, index, data, label,
                 transform=None, target_transform=None):
        super(Dummy, self).__init__(index, transform=transform,
                                    target_transform=target_transform)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert('RGB')
        target = self.label

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)