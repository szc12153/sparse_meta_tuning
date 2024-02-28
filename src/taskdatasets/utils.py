import numpy as np
import math
import torch
import random
import torch.nn.functional as F

def shuffle(images, labels):
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]


def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class TaskResampler:
    def __init__(self, images, labels, batch_size, device, target_images=None, target_labels=None):
        self.device = device
        self.context_images = None
        self.context_labels = None
        self.batch_size = batch_size
        self.target_images = target_images
        self.target_labels = target_labels
        self.num_target_batches = 0
        self.target_set_size = 0
        if (target_images is not None) and (target_labels is not None):
            # target set has been supplied
            self.target_set_size = len(target_labels)
            self.context_images = images
            self.context_labels = labels
        else:
            # no target set supplied, so need to split the supplied images into context and target
            self.context_images, self.context_labels, self.target_images, self.target_labels = \
                self._split(images, labels)
        unique_classes, class_counts = torch.unique(self.context_labels, sorted=True, return_counts=True)
        self.num_classes = len(unique_classes)
        self.class_counts = class_counts.cpu().numpy()
        self.classes = unique_classes.cpu().numpy()
        self.min_classes = 5
        self.max_target_size = 2000  # this was 2000 for VTAB
        _, target_class_counts = torch.unique(self.target_labels, sorted=True, return_counts=True)
        self.target_class_counts = target_class_counts.cpu().numpy()

    def get_task(self):
            return self._resample_task()

    def _split(self, images, labels):
        # split into context and target as close to 50/50 as possible
        # if an odd number of samples, give more to the context
        # if only one in a class, this will not work!
        context_images = []
        context_labels = []
        target_images = []
        target_labels = []

        # loop through classes and assign
        classes, class_counts = torch.unique(labels, sorted=True, return_counts=True)
        for cls in classes:
            class_count = class_counts[cls]
            context_count = math.ceil(class_count / 2.0)
            class_images = torch.index_select(images, 0, extract_class_indices(labels, cls))
            class_labels = torch.index_select(labels, 0, extract_class_indices(labels, cls))
            context_images.append(class_images[:context_count])
            context_labels.append(class_labels[:context_count])
            target_images.append(class_images[context_count:])
            target_labels.append(class_labels[context_count:])
        context_images = torch.vstack(context_images)
        context_labels = torch.hstack(context_labels)
        target_images = torch.vstack(target_images)
        target_labels = torch.hstack(target_labels)

        return context_images, context_labels, target_images, target_labels

    def _resample_task(self):
        context_batch_images = []
        context_batch_labels = []
        target_batch_images = []
        target_batch_labels = []

        # choose a random number of classes
        min_classes = min(self.num_classes, self.min_classes)
        max_classes = min(self.num_classes, self.batch_size)
        way = np.random.randint(min_classes, max_classes + 1)
        selected_classes = np.random.choice(self.classes, size=way, replace=False)

        # TODO also in the case of 1 shot, may not be a matching target class
        balanced_shots_to_use = max(round(float(self.batch_size) / float(len(selected_classes))), 1)
        for index, cls in enumerate(selected_classes):
            # resample a new context set
            num_shots_in_class = self.class_counts[np.where(self.classes == cls)[0][0]]
            num_shots_to_use = min(num_shots_in_class, balanced_shots_to_use)
            selected_shots = torch.randperm(num_shots_in_class, device=self.context_images.device)[:num_shots_to_use]
            context_class_images = torch.index_select(self.context_images, 0, extract_class_indices(self.context_labels, cls))
            context_class_labels = torch.index_select(self.context_labels, 0, extract_class_indices(self.context_labels, cls))
            selected_class_images = torch.index_select(context_class_images, 0, selected_shots)
            selected_class_labels = torch.index_select(context_class_labels, 0, selected_shots)
            selected_class_labels = selected_class_labels.fill_(index)
            context_batch_images.append(selected_class_images)
            context_batch_labels.append(selected_class_labels)

            # resample a new target set using the same classes
            max_target_shots = max(1, self.max_target_size // way)
            num_target_shots_in_class = self.target_class_counts[np.where(self.classes == cls)[0][0]]
            num_shots_to_use = min(num_target_shots_in_class, max_target_shots)
            selected_shots = torch.randperm(num_target_shots_in_class, device=self.target_images.device)[:num_shots_to_use]
            all_target_class_images = torch.index_select(self.target_images, 0, extract_class_indices(self.target_labels, cls))
            all_target_class_labels = torch.index_select(self.target_labels, 0, extract_class_indices(self.target_labels, cls))
            target_class_images = torch.index_select(all_target_class_images, 0, selected_shots)
            target_class_labels = torch.index_select(all_target_class_labels, 0, selected_shots)

            target_class_labels = target_class_labels.fill_(index)
            target_batch_images.append(target_class_images)
            target_batch_labels.append(target_class_labels)

        context_batch_images = torch.vstack(context_batch_images)
        context_batch_labels = torch.hstack(context_batch_labels)
        context_batch_images, context_batch_labels = shuffle(context_batch_images, context_batch_labels)

        target_batch_images = torch.vstack(target_batch_images)
        target_batch_labels = torch.hstack(target_batch_labels)
        target_batch_images, target_batch_labels = shuffle(target_batch_images, target_batch_labels)

        # move the task to the device
        context_batch_images = context_batch_images.to(self.device)
        target_batch_images = target_batch_images.to(self.device)
        context_batch_labels = context_batch_labels.to(self.device)
        target_batch_labels = target_batch_labels.type(torch.LongTensor).to(self.device)

        return context_batch_images, context_batch_labels, target_batch_images, target_batch_labels


def random_hflip(tensor, prob):
    if prob > random.random():
        return tensor
    return torch.flip(tensor, dims=(3,))


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_offset(x, ratio=1, ratio_h=1, ratio_v=1):
    w, h = x.size(2), x.size(3)

    imgs = []
    for img in x.unbind(dim = 0):
        max_h = int(w * ratio * ratio_h)
        max_v = int(h * ratio * ratio_v)

        value_h = random.randint(0, max_h) * 2 - max_h
        value_v = random.randint(0, max_v) * 2 - max_v

        if abs(value_h) > 0:
            img = torch.roll(img, value_h, 2)

        if abs(value_v) > 0:
            img = torch.roll(img, value_v, 1)

        imgs.append(img)

    return torch.stack(imgs)


def rand_offset_h(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=ratio, ratio_v=0)


def rand_offset_v(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=0, ratio_v=ratio)


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


class DiffAugment:
    augment_funcs = {
        'color': [rand_brightness, rand_saturation, rand_contrast],
        'offset': [rand_offset],
        'offset_h': [rand_offset_h],
        'offset_v': [rand_offset_v],
        'translation': [rand_translation],
        'cutout': [rand_cutout],
    }

    def __init__(self, types, prob=0.5, detach=True):
        self.types = types
        self.prob = prob
        self.detach = detach

    def __call__(self, x):
        if random.random() < self.prob:
            with torch.set_grad_enabled(not self.detach):
                x = random_hflip(x, prob=0.5)
                for p in self.types:
                    for f in self.augment_funcs[p]:
                        x = f(x)
                x = x.contiguous()
        return x





from collections import OrderedDict, defaultdict
from torchmeta.utils.data.task import Task, ConcatTask, SubsetTask
from torchmeta.transforms.utils import apply_wrapper

class Splitter(object):
    def __init__(self, splits, random_state_seed):
        self.splits = splits
        self.random_state_seed = random_state_seed
        self.seed(random_state_seed)

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed=seed)

    def get_indices(self, task):
        if isinstance(task, ConcatTask):
            indices = self.get_indices_concattask(task)
        elif isinstance(task, Task):
            indices = self.get_indices_task(task)
        else:
            raise ValueError('The task must be of type `ConcatTask` or `Task`, '
                'Got type `{0}`.'.format(type(task)))
        return indices

    def get_indices_task(self, task):
        raise NotImplementedError('Method `get_indices_task` must be '
            'implemented in classes inherited from `Splitter`.')

    def get_indices_concattask(self, task):
        raise NotImplementedError('Method `get_indices_concattask` must be '
            'implemented in classes inherited from `Splitter`.')

    def _get_class_indices(self, task):
        class_indices = defaultdict(list)
        if task.num_classes is None: # Regression task
            class_indices['regression'] = range(len(task))
        else:
            for index in range(len(task)):
                sample = task[index]
                if (not isinstance(sample, tuple)) or (len(sample) < 2):
                    raise ValueError('In order to split the dataset in train/'
                        'test splits, `Splitter` must access the targets. Each '
                        'sample from a task must be a tuple with at least 2 '
                        'elements, with the last one being the target.')
                class_indices[sample[-1]].append(index)

            if len(class_indices) != task.num_classes:
                raise ValueError('The number of classes detected in `Splitter` '
                    '({0}) is different from the property `num_classes` ({1}) '
                    'in task `{2}`.'.format(len(class_indices),
                    task.num_classes, task))

        return class_indices

    def __call__(self, task):
        indices = self.get_indices(task)
        return OrderedDict([(split, SubsetTask(task, indices[split]))
            for split in self.splits])

    def __len__(self):
        return len(self.splits)


class ClassSplitter_(Splitter):
    def __init__(self, shuffle=True, num_samples_per_class=None,
                 num_train_per_class=None, num_test_per_class=None,
                 num_support_per_class=None, num_query_per_class=None,
                 random_state_seed=0):
        """
        Transforms a dataset into train/test splits for few-shot learning tasks,
        based on a fixed number of samples per class for each split. This is a
        dataset transformation to be applied as a `dataset_transform` in a
        `MetaDataset`.

        Parameters
        ----------
        shuffle : bool (default: `True`)
            Shuffle the data in the dataset before the split.

        num_samples_per_class : dict, optional
            Dictionary containing the names of the splits (as keys) and the
            corresponding number of samples per class in each split (as values).
            If not `None`, then the arguments `num_train_per_class`,
            `num_test_per_class`, `num_support_per_class` and
            `num_query_per_class` are ignored.

        num_train_per_class : int, optional
            Number of samples per class in the training split. This corresponds
            to the number of "shots" in "k-shot learning". If not `None`, this
            creates an item `train` for each task.

        num_test_per_class : int, optional
            Number of samples per class in the test split. If not `None`, this
            creates an item `test` for each task.

        num_support_per_class : int, optional
            Alias for `num_train_per_class`. If `num_train_per_class` is not
            `None`, then this argument is ignored. If not `None`, this creates
            an item `support` for each task.

        num_query_per_class : int, optional
            Alias for `num_test_per_class`. If `num_test_per_class` is not
            `None`, then this argument is ignored. If not `None`, this creates
            an item `query` for each task.

        random_state_seed : int, optional
            seed of the np.RandomState. Defaults to '0'.

        Examples
        --------
        >>> transform = ClassSplitter(num_samples_per_class={
        ...     'train': 5, 'test': 15})
        >>> dataset = Omniglot('data', num_classes_per_task=5,
        ...                    dataset_transform=transform, meta_train=True)
        >>> task = dataset.sample_task()
        >>> task.keys()
        ['train', 'test']
        >>> len(task['train']), len(task['test'])
        (25, 75)
        """
        self.shuffle = shuffle

        if num_samples_per_class is None:
            num_samples_per_class = OrderedDict()
            if num_train_per_class is not None:
                num_samples_per_class['train'] = num_train_per_class
            elif num_support_per_class is not None:
                num_samples_per_class['support'] = num_support_per_class
            if num_test_per_class is not None:
                num_samples_per_class['test'] = num_test_per_class
            elif num_query_per_class is not None:
                num_samples_per_class['query'] = num_query_per_class
        assert len(num_samples_per_class) > 0

        self._min_samples_per_class = sum(num_samples_per_class.values())
        super(ClassSplitter_, self).__init__(num_samples_per_class, random_state_seed)

    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
        indices = OrderedDict([(split, []) for split in self.splits])

        for i, (name, class_indices) in enumerate(all_class_indices.items()):
            num_samples = len(class_indices)
            if num_samples < self._min_samples_per_class:
                raise ValueError('The number of samples for class `{0}` ({1}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({2}).'.format(name,
                    num_samples, self._min_samples_per_class))

            if self.shuffle:
                seed = (hash(task) + i + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples)
            else:
                dataset_indices = np.arange(num_samples)

            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle and split != "test":
                    self.np_random.shuffle(split_indices)
                indices[split].extend([class_indices[idx] for idx in split_indices])
                ptr += num_split

        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0

        for dataset in task.datasets:
            num_samples = len(dataset)
            if num_samples < self._min_samples_per_class:
                # raise ValueError('The number of samples for one class ({0}) '
                #     'is smaller than the minimum number of samples per class '
                #     'required by `ClassSplitter` ({1}).'.format(num_samples,
                #     self._min_samples_per_class))
                print(f"up sample {self._min_samples_per_class - num_samples} images for the current task")
                # # insteaad of rasing an error: up sample the class
                # seed = (hash(task) + hash(dataset) + self.random_state_seed) % (2 ** 32)
                # dataset_indices = np.arange(num_samples)
                # np.concatenate([dataset_indices, ])
                up_sample = self._min_samples_per_class - num_samples


            else:
                up_sample = 0

            if self.shuffle:
                seed = (hash(task) + hash(dataset) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples)
            else:
                dataset_indices = np.arange(num_samples)
                seed = 0

            if up_sample:
                dataset_indices = np.concatenate([dataset_indices, np.random.RandomState(seed).choice(num_samples, size =up_sample)])

            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle and split != "test":
                    self.np_random.shuffle(split_indices)
                indices[split].extend(split_indices + cum_size)
                ptr += num_split
            cum_size += num_samples

        return indices


def ClassSplitter(task=None, *args, **kwargs):
    return apply_wrapper(ClassSplitter_(*args, **kwargs), task)

