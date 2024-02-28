import os
import torch
import pickle
import shutil
import yaml
from dataclasses import dataclass, asdict
from dacite import from_dict
import logging
from .configurations import Configuration
import random
import sys
import logging
import math
import numpy as np


def set_random_seed(seed, deterministic=True):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_path(path):
    if os.path.exists(path) and input(f"overwrite existing dir [y/Any] : {path} ?")!="y":
        sys.exit('Rename run-name and initiate new run')
    else:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        os.makedirs(path)
    return


def load_config_from_yaml(file_path: str, safe_load: bool = True) -> dataclass:
    with open(file_path) as f:
        cfg = yaml.safe_load(f) if safe_load else yaml.load(f, Loader=yaml.CLoader)  # config is dict
    print('Reloaded exp configuration from file : {}'.format(file_path))
    cfg = from_dict(data_class=Configuration, data=cfg)
    print(yaml.dump(asdict(cfg), default_flow_style=False))
    return cfg


def save_config_to_yaml(cfg: dataclass, dir_path: str) -> None:
    cfg_dict = asdict(cfg)
    ## delete useless args before save
    useless_key = []
    for key in cfg_dict:
        if cfg_dict[key] is None:
            useless_key.append(key)
    for key in useless_key:
        del cfg_dict[key]
    with open(dir_path, 'w') as outfile:
        yaml.dump(cfg_dict, outfile, default_flow_style=False)
    print('Saved configuration to directory : {}'.format(dir_path))


def save_all_py_files_in_src(source_path, destination_path, override=True):
    """
    Recursive copies files from source  to destination directory.
    :param source_path: source directory
    :param destination_path: destination directory
    :param override if True all files will be overridden otherwise skip if file exist
    :return: count of copied files
    """
    import glob
    import shutil
    files_count = 0
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    items = glob.glob(source_path+'/*')
    for item in items:
        if os.path.isdir(item):
            path = os.path.join(destination_path, item.split('/')[-1])
            files_count += save_all_py_files_in_src(source_path=item,destination_path=path, override=override)
        else:
            if item.endswith('.py'):
                file = os.path.join(destination_path, item.split('/')[-1])
                if not os.path.exists(file) or override:
                    shutil.copyfile(item, file)
                    files_count += 1
    return files_count


def get_logger(name, display_level="debug"):

    class CustomFormatter(logging.Formatter):
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        format = "'%(name)-12s: %(levelname)-8s %(message)s'"

        FORMATS = {
            logging.DEBUG: grey + format + reset,
            logging.INFO: grey + format + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: bold_red + format + reset
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    level = {"info":logging.INFO,"debug":logging.DEBUG,"warning":logging.WARNING,"error":logging.ERROR}
    console = logging.StreamHandler()
    console.setLevel(level[display_level])
    console.setFormatter(CustomFormatter())
    logger = logging.getLogger(name)
    logger.addHandler(console)
    return logger



# Metrics

# Some keys used for the following dictionaries
COUNT = "count"
CONF = "conf"
ACC = "acc"
BIN_ACC = "bin_acc"
BIN_CONF = "bin_conf"

def accuracy(predictions, targets)->torch.Tensor:
    predictions = predictions.argmax(dim=-1).view(targets.shape)
    return (predictions == targets).float().sum() / targets.size(0)


def is_better(curr:dict,curr_best:dict)->bool:
    if curr_best is None:
        return True
    return np.mean(list(curr.values())) > np.mean(list(curr_best.values()))


def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if bin_dict[binn][COUNT] == 0:
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(
                bin_dict[binn][COUNT]
            )
    return bin_dict


def expected_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
    return ece


def maximum_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    ce = []
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        ce.append(abs(bin_accuracy - bin_confidence))
    return max(ce)


def average_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    non_empty_bins = 0
    ace = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        if bin_count > 0:
            non_empty_bins += 1
        ace += abs(bin_accuracy - bin_confidence)
    return ace / float(non_empty_bins)

