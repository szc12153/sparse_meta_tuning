import os
import h5py
from PIL import Image

import torch


class MetaValDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, num_episodes=4000, return_source_info= False, source_label=None):
        super().__init__()

        self.num_episodes = num_episodes
        self.h5_path = h5_path
        self.h5_file = None
        self.return_source_info = return_source_info
        self.source_label = source_label

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        record = self.h5_file[str(idx)]
        support_images = record['sx'][()]
        support_labels = record['sy'][()]
        query_images = record['x'][()]
        query_labels = record['y'][()]

        if self.return_source_info:
            source_label = self.source_label
            return support_images, support_labels, query_images, query_labels, source_label
        else:
            return support_images, support_labels, query_images, query_labels