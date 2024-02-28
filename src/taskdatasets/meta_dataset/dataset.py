import os
import numpy as np
from .meta_val_dataset import MetaValDataset
from .meta_h5_dataset import FullMetaDatasetH5
from .meta_dataset.utils import Split
from src.utils import get_logger

logger = get_logger(__name__)

class META_DATASET_ARGS:
    data_path = os.getenv("mdh5_path")
    assert os.path.isdir(data_path)
    shuffle = True
    image_size = 128
    test_transforms = ['resize', 'center_crop', 'to_tensor', 'normalize']
    train_transforms = ['random_resized_crop', 'jitter', 'random_flip', 'to_tensor', 'normalize']
    num_ways = None
    num_support = None
    num_query = None
    min_ways = 5
    max_ways_upper_bound = 50
    max_num_query = 10
    max_support_set_size = 500
    max_support_size_contrib_per_class= 100
    min_log_weight = np.log(0.5)
    max_log_weight = np.log(2)
    ignore_dag_ontology = False
    ignore_bilevel_ontology = False
    ignore_hierarchy_probability = 0.0
    min_examples_in_class = 0
    nEpisode = 2000
    nValEpisode = 120
    nTestEpisode = 600
    dataset_to_idx = ["ilsvrc_2012", "omniglot", 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower',
                      'traffic_sign', 'mscoco']
    base_sources = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower']
    val_sources =  ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower']
    test_sources = ['aircraft', 'cu_birds', 'ilsvrc_2012', 'omniglot',  'dtd', 'quickdraw', 'fungi', 'vgg_flower','traffic_sign', 'mscoco']
    return_source_info = False
    def __init__(self,**kwargs):
        for n,v in kwargs.items():
            if v is not None:
                if not hasattr(self,n):
                    logger.warn(f"META_DATASET_ARGS does not have attribute :{n}")
                    continue
                setattr(self,n,v)
                logger.info(f"force reset {n} to {v}")


def MetaDataset(split, **kwargs)->dict:
    md_configs = META_DATASET_ARGS(**kwargs)
    if split == "test":
        testSet={}
        all_test_sources = md_configs.test_sources
        for source in all_test_sources:
            md_configs.test_sources=[source]
            testSet[source] = FullMetaDatasetH5(md_configs, Split.TEST)
        return testSet
    if split =="val":
        valSet = {}
        for source in md_configs.val_sources:
            try:
                source_label = md_configs.dataset_to_idx.index(source)
            except ValueError:
                source_label = -1
            valSet[source] = MetaValDataset(os.path.join(md_configs.data_path, source, f'val_ep{md_configs.nValEpisode}_img{md_configs.image_size}.h5'),
                                            num_episodes=md_configs.nValEpisode,
                                            return_source_info = md_configs.return_source_info,
                                            source_label = source_label)
        return valSet
    if split=="train":
        return {"single" : FullMetaDatasetH5(md_configs, Split.TRAIN)}
    raise Exception(f"unknown split {split} for meta-dataset")