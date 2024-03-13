from .domainnet import DomainnetTuneDataset
from .cars import CARSDataset
from .meta_dataset import MetaDataset
from .pets import PetDataset
from .food import FoodDataset
from .utils import TaskResampler, DiffAugment, ClassSplitter
from torchmeta.transforms import Categorical
from torchmeta.utils.data import BatchMetaDataLoader
from torch.utils.data import DataLoader
import torch
import random
import numpy as np


AVAILABLE_FEWSHOT_DATASETS={
    "pet": PetDataset,
    "food": FoodDataset,
    "sketch" :DomainnetTuneDataset,
    "real": DomainnetTuneDataset,
    "cars": CARSDataset,
    "infograph": DomainnetTuneDataset,
    "clipart": DomainnetTuneDataset,
    "quickdraw" : DomainnetTuneDataset,
    "painting": DomainnetTuneDataset,
}

AVAILABLE_FEWSHOT_DATASETS_PATH={
    "cars": None,
    "pet": None,
    "food": None,
    "country": None,
    "sketch" : None,
    "real": None,
    "infograph": None,
    "clipart": None,
    "quickdraw" : None,
    "painting": None,
}

def get_fewshot_taskloaders(dataset_names, split, input_size, n_ways,k_shots,k_queries, batch_size, train_val_shots=-1, aug=False)->dict:
    meta_dataloaders={}
    for name in dataset_names:
        dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=k_shots,
                                      num_test_per_class=k_queries)
        dataset = AVAILABLE_FEWSHOT_DATASETS[name]
        task_dataset = dataset(root=AVAILABLE_FEWSHOT_DATASETS_PATH[name],
                               meta_split=split,
                               transform=dataset.get_transform(split,input_size, aug=aug),
                               target_transform=Categorical(n_ways),
                               num_classes_per_task=n_ways,
                               dataset_transform=dataset_transform,
                               download=False)


        meta_dataloaders[name] = BatchMetaDataLoader(task_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     num_workers=4,
                                                     pin_memory=True)

    return meta_dataloaders


def get_metadataset_taskloaders(split, **kwargs)->dict:
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    meta_dataloaders={}
    task_datasets = MetaDataset(split, **kwargs)  # a dict of datasets (by domains for val and a single for train/test)
    for j, (source, task_dataset) in enumerate(task_datasets.items()):

        if split =="train":
            sampler = torch.utils.data.RandomSampler(task_dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(task_dataset)

        generator = torch.Generator()
        generator.manual_seed(10000 + j)

        meta_dataloaders[source] = torch.utils.data.DataLoader(
            task_dataset,
            sampler=sampler,
            batch_size=1,
            num_workers=4,  # more workers can take too much CPU
            pin_memory=True,
            drop_last=split=="train",
            worker_init_fn=worker_init_fn,
            generator=generator
        )

    return meta_dataloaders
