import math
import os.path
import random
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, ColorJitter, RandomRotation, RandomResizedCrop, RandomCrop
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from src.utils import get_logger


class DomainnetCraftDataset(ImageFolder):
    def __init__(self, root,domain,split,transform,target_transform,train_val_shots):
        loader = DomainnetCraftDataset.load_from_path
        super(DomainnetCraftDataset, self).__init__(os.path.join(root, domain),
                                                    transform,
                                                    target_transform,
                                                    loader=loader,
                                                    is_valid_file=None)
        self.get_split_samples(root, domain, split, train_val_shots)

    def get_split_samples(self, root, domain, split, max_n_shots):
        split_samples=[]
        split_file_path = os.path.join(root, "splits",f'{domain}_{"test" if split=="test" else "train"}.txt')
        with open(split_file_path,"r") as f:
            lines = f.read().splitlines()
            for l in lines:
                path, label = l.split(" ")
                split_samples.append((os.path.join(root,path), int(label)))

        print("before filtering :", len(split_samples))

        if split in ["train","valid"]:
            train_samples, val_samples =[], []
            labels = np.unique([sample[1] for sample in split_samples])
            for c in labels:
                c_samples = list(filter(lambda x: x[1] == c, split_samples))
                random.Random(c).shuffle(c_samples)
                split_idx = math.floor(len(c_samples) * 0.8)
                train_stop_idx = min(max_n_shots, split_idx) if max_n_shots!=-1 else split_idx
                train_samples.extend(c_samples[:train_stop_idx])
                val_stop_idx = min(split_idx+max_n_shots, len(c_samples)) if max_n_shots!=-1 else len(c_samples)
                val_samples.extend(c_samples[split_idx:val_stop_idx])

        if split=="train":
            self.samples = train_samples
        elif split=="valid":
            self.samples =val_samples
        elif split=="test":
            self.samples = split_samples
        else:
            raise AttributeError(split)
        print("after filtering :", len(self.samples))

    @staticmethod
    def load_from_path(path):
        return Image.open(path).convert("RGB")


    @staticmethod
    def get_transform(split, input_size, aug):
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





class DomainnetClassDataset(ClassDataset):
    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        _meta_split = "val" if meta_split=="valid" else meta_split
        super(DomainnetClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=_meta_split,
            class_augmentations=class_augmentations)
        self.transform = transform
        if meta_split=="val":
            meta_split="valid"
        self.root = root
        # self.root = os.path.join(root,meta_split)
        # collect all classes
        self.data={}
        self.labels=[]
        # randomly select training classes for train and val:
        split_classes = self._get_split_classes(meta_split)
        for c, c_names in enumerate(os.listdir(self.root)):
            if c_names not in split_classes:
                # some domain may not have the full list of classes
                continue
            c_root = os.path.join(self.root, c_names)
            imgs = [os.path.join(self.root, c_names, p) for p in os.listdir(c_root)]
            # 2/8 split for val and test ( No training)
            shuffled_idx = np.random.default_rng(seed=c).permutation(len(imgs))
            #val_idx, test_idx  = shuffled_idx[:int(len(imgs)*0.2)], shuffled_idx[int(len(imgs)*0.2):]
            #if meta_split=="valid":
            #    split_imgs = [imgs[idx] for idx in val_idx]
            #else:
            #    split_imgs = [imgs[idx] for idx in test_idx]
            repeat_idx = 0
            split_imgs = [imgs[idx] for idx in shuffled_idx]
            while len(split_imgs) < 20:
                split_imgs.append(split_imgs[repeat_idx])  # up sample images
                repeat_idx += 1
            self.data[c] = split_imgs
            self.labels.append(c)
        print(f"using {len(self.labels)} classes for split {meta_split}")

    def _get_split_classes(self, split):
        try:
            with open(f"src/taskdatasets/domainnet/splits.json", "r") as f:
                splits = json.load(f)
        except FileNotFoundError:
            all_classes = os.listdir(self.root)
            random.shuffle(all_classes)
            tv_split_idx,vt_split_idx = math.floor(len(all_classes)*0.6), math.floor(len(all_classes)*0.75)
            splits={
                "train" : all_classes[:tv_split_idx],
                "valid": all_classes[tv_split_idx:vt_split_idx],
                "test":all_classes[vt_split_idx:]
            }
            with open(f"src/taskdatasets/domainnet/splits.json", "w+") as outfile:
                json.dump(splits, outfile)
        return splits[split]
        # if split =="valid":
        #     return splits["valid"]
        # elif split in ["train","test"]:
        #     return splits["train"] + splits["test"]
        # else:
        #     raise KeyError

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return Dummy(index, data, label, transform=transform,
                          target_transform=target_transform)

    @property
    def num_classes(self):
        return len(self.labels)


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


class DomainnetTuneDataset(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = DomainnetClassDataset(root, meta_train=meta_train, meta_val=meta_val,
                                  meta_test=meta_test, meta_split=meta_split, transform=transform,
                                  class_augmentations=class_augmentations, download=download)
        super(DomainnetTuneDataset, self).__init__(dataset, num_classes_per_task,
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


class DomainnetOODQueryClassDataset(ClassDataset):
    def __init__(self, root, sq_datasets,n_shots, n_queries, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(DomainnetOODQueryClassDataset, self).__init__(meta_train=meta_train,
                                                    meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                                    class_augmentations=class_augmentations)
        self._random_seed = 0
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.transform = transform
        if meta_split == "val":
            meta_split = "valid"
        self.root= {}
        self.data={}
        self.labels={}
        self.support_domain, self.query_domain= sq_datasets.split("->")
        for domain, min_samples_req in zip([self.support_domain, self.query_domain],[self.n_shots, self.n_queries]):
            assert domain in ["real","clipart","sketch","infograph","quickdraw","painting"]
            # if domain =="quickdraw":
            #     self.root[domain]= os.path.join(root, "quickdraw_dn",meta_split)
            # else:
            self.root[domain] = os.path.join(root, domain)
            # collect all classes
            self.data[domain] = {}
            self.labels[domain]=[]
            # randomly select training classes for train and val:
            split_classes = self._get_split_classes(meta_split)
            avaliable_classes = os.listdir(self.root[domain])


            for global_c_idx, c_names in enumerate(split_classes):
                if c_names not in avaliable_classes:
                    # some domain may not have the full list of classes
                    continue
                c_root = os.path.join(self.root[domain], c_names)
                imgs = [os.path.join(self.root[domain], c_names, p) for p in os.listdir(c_root)]

                shuffled_idx = np.random.default_rng(seed=global_c_idx).permutation(len(imgs))
                #val_idx, test_idx  = shuffled_idx[:int(len(imgs)*0.2)], shuffled_idx[int(len(imgs)*0.2):]
                #if meta_split=="valid":
                #    split_imgs = [imgs[idx] for idx in val_idx]
                #else:
                #    split_imgs = [imgs[idx] for idx in test_idx]
                split_imgs = [imgs[idx] for idx in shuffled_idx]
                repeat_idx = 0
                while len(split_imgs) < 20:
                    split_imgs.append(split_imgs[repeat_idx])  # up sample images
                    repeat_idx += 1
                self.data[domain][global_c_idx] = split_imgs
                self.labels[domain].append(global_c_idx)
        # remove unique classes in query and support,
        #for query_c_idx in self.labels[self.query_domain]:
        #    if self.data[self.support_domain].get(query_c_idx) is None:
        #        self.data[self.query_domain].pop(query_c_idx)
        #for support_c_idx in self.labels[self.support_domain]:
        #    if self.data[self.query_domain].get(support_c_idx) is None:
        #        self.data[self.support_domain].pop(support_c_idx)
        assert len(self.data[self.query_domain])==len(self.data[self.support_domain])
        self.labels = list(set(self.labels[self.support_domain]+self.labels[self.query_domain]))
        print(f"{self.support_domain}->{self.query_domain} using {len(self.labels)} classes for split {meta_split}")

    def _get_split_classes(self, split):
        try:
            with open(f"src/taskdatasets/domainnet/splits.json", "r") as f:
                splits = json.load(f)
            assert sum(len(v) for v in splits.values())==345
        except FileNotFoundError:
            all_classes = os.listdir(self.root)
            random.shuffle(all_classes)
            tv_split_idx, vt_split_idx = math.floor(len(all_classes) * 0.6), math.floor(len(all_classes) * 0.75)
            splits = {
                "train": all_classes[:tv_split_idx],
                "valid": all_classes[tv_split_idx:vt_split_idx],
                "test": all_classes[vt_split_idx:]
            }
            with open(f"src/taskdatasets/domainnet/splits.json", "w+") as outfile:
                json.dump(splits, outfile)
        return splits[split]
        # if split =="valid":
        #     return splits["valid"]
        # else:
        #     return splits["train"] + splits["test"]

    def __getitem__(self, index):
        #random.seed(self._random_seed)
        label = self.labels[index % self.num_classes]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        support_data = random.sample(self.data[self.support_domain][label], self.n_shots)
        query_data = random.sample(self.data[self.query_domain][label], self.n_queries)
        data = support_data + query_data
        #self._random_seed += 1
        return Dummy(index, data, label, transform=transform,
                     target_transform=target_transform)

    @property
    def num_classes(self):
        return len(self.labels)


class DomainnetOODQueryDataset(DomainnetTuneDataset):
    def __init__(self, root, sq_datasets, n_shots, n_queries, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):

        dataset = DomainnetOODQueryClassDataset(root, sq_datasets, n_shots, n_queries,  meta_train=meta_train, meta_val=meta_val,
                                        meta_test=meta_test, meta_split=meta_split, transform=transform,
                                        class_augmentations=class_augmentations, download=download)
        super(DomainnetTuneDataset, self).__init__(dataset, num_classes_per_task,
                                                   target_transform=target_transform,
                                                   dataset_transform=dataset_transform)

