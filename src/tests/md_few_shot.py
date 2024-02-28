import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from src.taskdatasets import get_metadataset_taskloaders, DiffAugment, get_fewshot_taskloaders
from src.utils import  Configuration, expected_calibration_error as ece, get_logger
import src.learners.base_learners.torchhub_vit as TorchHubViTModels
from src.learners  import get_meta_learner, get_base_learner
from .finetuners import get_finetune_model
from collections import defaultdict, namedtuple
import numpy as np
from tqdm import trange, tqdm
from src.plots.ada_trajectory import plt_and_save_adaptation_trajectory, save_raw_pkl
from copy import deepcopy
from typing import Union, List


logger = get_logger(__name__)


class FewShotDataManager:
    pbar = None
    get_datalaoder = None
    name = None
    @staticmethod
    def set_pbar_description(message):
        FewShotDataManager.pbar.set_description(message)

    @staticmethod
    def tasks_in_epoch(task_loaders, device):
        if isinstance(task_loaders, dict):
            FewShotDataManager.pbar = trange(500, leave=True)
            for i in FewShotDataManager.pbar:
                for source_id, train_loader in enumerate(task_loaders.items()):
                    batch = next(train_loader)
                    for task in zip(*batch["train"], *batch["test"]):
                        yield [_.to(device) for _ in task] + [torch.tensor([-1]).to(device)]# x_s,y_s,x_q,y_q
        elif isinstance(task_loaders, torch.utils.data.DataLoader):  # evaluate and test
            FewShotDataManager.pbar = tqdm(task_loaders)
            for batch in FewShotDataManager.pbar:
                for task in zip(*batch["train"], *batch["test"]):
                    yield [_.to(device) for _ in task]  + [torch.tensor([-1]).to(device)] # x_s,y_s,x_q,y_q

    @staticmethod
    def get_split_loaders(split, args, **kwargs):
        if split=="val":
             return FewShotDataManager.get_datalaoder(dataset_names=args.test.datasets,
                                                      split="val",
                                                      input_size=224,
                                                      n_ways=args.test.num_ways,
                                                      k_shots=args.test.num_shots,
                                                      k_queries=args.test.num_queries,
                                                      batch_size=1)
        if split=="test":
            return FewShotDataManager.get_datalaoder(dataset_names=args.test.datasets,
                                                     split="test",  ## todo: change this back to test
                                                     input_size=224,
                                                     n_ways=args.test.num_ways,
                                                     k_shots=args.test.num_shots,
                                                     k_queries=args.test.num_queries,
                                                     batch_size=1)
        raise KeyError(f"undefined split {split} for few-shot dataset")


class MetaDatasetDatamanager:
    name="md"
    pbar = None

    @staticmethod
    def set_pbar_description(message):
        MetaDatasetDatamanager.pbar.set_description(message)

    @staticmethod
    def tasks_in_epoch(task_loaders, device):
        """2000 tasks per domain in an epoch"""
        if isinstance(task_loaders, dict):
            MetaDatasetDatamanager.pbar = tqdm(task_loaders["single"], leave=True)  # train and test
        elif isinstance(task_loaders, torch.utils.data.DataLoader):  # evaluate and test
            MetaDatasetDatamanager.pbar = tqdm(task_loaders, leave=True)
        for task in MetaDatasetDatamanager.pbar :
            yield [_[0].to(device) for _ in task]

    @staticmethod
    def get_split_loaders(split, args, **kwargs):
        return get_metadataset_taskloaders(split=split, **kwargs)


class MetaTester:
    def __init__(self, args: Configuration, DataManager:Union[FewShotDataManager,MetaDatasetDatamanager], checkpoint:str):
        self.device = "cuda:0"
        """pmf"""
        if args.alg == "pmf-full": ## pmf pretrained + full-fine-tuning
            backbone = TorchHubViTModels.__dict__['vit_small'](patch_size=16, num_classes=0)
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            backbone.load_state_dict(state_dict, strict=True)  # load pre-trained state dict
            self.learner = get_finetune_model(finetune="full")(net=backbone, args=args).to(self.device)
            self.load_checkpoint(checkpoint) # load pmf state dict
        elif args.alg == "pmf-lora": ## pmf pretrained + lora fine-tuning 
            from transformers import ViTModel
            backbone = ViTModel.from_pretrained('WinKawaks/vit-small-patch16-224')
            backbone.pooler = None
            self.learner = get_finetune_model(finetune="lora")(net=backbone, args=args).to(self.device)
            # call this so that the pre-traiened state dict are saved by the learner
            if checkpoint:
                self.load_checkpoint(checkpoint)
            else:
                pre_trained_state_dict = deepcopy(self.learner.state_dict())
                self.learner.load_state_dict(pre_trained_state_dict)
        elif args.alg == "pretrained": ## use pre-trained backbone,
            args.base_learner.backbone ="ViT_Small_DINO"
            logger.info(args.base_learner.backbone)
            backbone = get_base_learner(args)
            self.learner = get_finetune_model(finetune="full")(net=backbone, args=args).to(self.device)
            # call this so that the pre-traiened state dict are saved by the learner
            pre_trained_state_dict = deepcopy(self.learner.state_dict())
            self.learner.load_state_dict(pre_trained_state_dict)
        elif args.alg=="smat":
            base_learner = get_base_learner(args)
            meta_learner = get_meta_learner(base_learner, args).to(self.device)
            self.learner = meta_learner
            self.load_checkpoint(checkpoint) # load and save meta-tuned state dict
            if "lora" in args.finetune_mode:
                self.learner.pretrained_state_dict = deepcopy(self.learner.net)

        self.args = args
        self.datasetroutine = DataManager

    def _load_cached_val_tasks(self)->dict:
        import pickle
        val_tasks = []
        for dataset in os.listdir("cached_val_tasks"):
            if dataset in ["infograph.pkl","real.pkl","sketch.pkl","painting.pkl","clipart.pkl"]:
                continue
            with open(os.path.join("cached_val_tasks", dataset), 'rb') as f:
                tasks = pickle.load(f)["tasks"]
            val_tasks.extend([t for i,t in enumerate(tasks) if i <5])
            logger.info(f"adding val split from {dataset}")

        for t in val_tasks:
            x_s,y_s = [],[]
            for x,y in zip(t[0],t[1]):
                if y_s.count(y) < self.args.md_oneshot:
                    x_s.append(x)
                    y_s.append(y)
            t[0],t[1] = torch.stack(x_s), torch.stack(y_s).reshape(-1)
        logger.info(f"truncate support to {self.args.md_oneshot} sample/class max")
        return val_tasks

    def load_checkpoint(self,path):
        assert os.path.isfile(path), path
        checkpoint = torch.load(path, map_location="cuda")
        self.learner.load_state_dict(checkpoint["model"], strict=True)
        logger.info(f"succesfully loaded MODEL state_dict from {path}")

    def evaluate_single_dataset(self, dataset, val_loader, test_loader, n_tasks, track_and_plot, finetune_mode, hps) -> namedtuple:
        def _get_mu_and_conf(traj, reduction_dim) -> (np.ndarray, np.ndarray):
            """
            Parameters
            ----------
            traj : n_tasks, n_steps

            Returns
            -------

            """
            if traj.dim() == 1:
                traj_mu = traj.cpu().numpy()
                traj_conf = np.zeros_like(traj_mu)
            elif traj.dim() == 2:
                traj_mu = traj.mean(reduction_dim).cpu().numpy()
                traj_conf = ((1.96 * traj.std(reduction_dim)) / math.sqrt(traj.size(reduction_dim))).cpu().numpy()
            else:
                raise Exception(traj.shape)
            return traj_mu, traj_conf

        model_train_state = self.learner.training
        logger.info(f"using {finetune_mode} mode for fine-tuning on the support set")
        self.learner.eval()
        Metrics = namedtuple('Metrics', track_and_plot)
        all_traj = defaultdict(list)
        diff_aug = DiffAugment(types=['color', 'translation'], prob=0.9, detach=True)

        if hps: ## follow pmf 
            val_tasks = []
            if val_loader:
                hps_loader = val_loader 
            else:
                hps_loader = test_loader
            max_val_tasks = 10 if val_loader is not None else 5
            for task_counter, task in enumerate(self.datasetroutine.tasks_in_epoch(hps_loader, device=self.device)):
                x_s, y_s = [], []
                for x, y in zip(task[0], task[1]):
                    if not self.args.md_oneshot or y_s.count(y) < self.args.md_oneshot:
                        x_s.append(x)
                        y_s.append(y)
                task[0], task[1] = torch.stack(x_s), torch.stack(y_s).reshape(-1)
                val_tasks.append(task)
                if task_counter + 1 >= max_val_tasks:
                    logger.info(f"truncate support to {self.args.md_oneshot} sample/class max")
                    break
            best_step = self.find_best_finetune_configs(val_tasks = val_tasks, dataset =dataset, aug_func=diff_aug if self.args.aug else None, finetune_mode = finetune_mode)
        else:
            best_step = -1

        # evaluation
        task_acc_mean = 0
        task_acc_max = 0
        all_raw_pred_prob =[]
        for task_counter, task in enumerate(self.datasetroutine.tasks_in_epoch(test_loader, device=self.device)):
            x_s, y_s, x_q, y_q = task[:4]

            source_label = y_q.new((1,)).fill_(-1)
            res_dict = self.learner(x_s=x_s,
                                    y_s=y_s,
                                    x_q=x_q,
                                    y_q=None,
                                    track_inner_trajectory = True,
                                    aug_func= diff_aug if self.args.aug else None,
                                    use_protomaml = self.args.use_protomaml,
                                    source_label = source_label,
                                    finetune_mode = finetune_mode )
            all_traj["y_q"].append(y_q.view(1, -1))  # 1, n_query
            step_y_q_pred = F.softmax(torch.stack(res_dict["y_q_pred"]), dim=-1)  # n_steps, n_query, pred_dim
            all_raw_pred_prob.append(step_y_q_pred)
            task_acc = (step_y_q_pred.argmax(dim=-1) == all_traj["y_q"][-1]).float().mean(dim=-1,keepdim=True) * 100
            task_acc_mean += task_acc # num_steps, 1
            task_acc_max += task_acc.max().item()
            logger.info(f"No.{task_counter:<3} ways: {y_s.max().item()+1} support: {x_s.size(0)}, "
                        f"end acc : {task_acc[-1].item():.3f}, "
                        f"best acc : {task_acc.max().item():.3f} @ {task_acc.argmax().item()} "
                        f"upper bound avg: {task_acc_max/(task_counter+1):.2f}") # 
            
            self.datasetroutine.set_pbar_description(f'{dataset:<30} Acc: {task_acc_mean[-1].item()/(task_counter+1):.2f} ' 
                                                     f'Max: {task_acc_mean.max().item()/(task_counter+1):.2f} @ {task_acc_mean.argmax().item()} '
                                                     f'U: {task_acc_max/(task_counter+1):.2f}')
            confs,pred = step_y_q_pred.max(dim=-1)
            all_traj["y_q_pred"].append(pred)  # n_steps, n_query
            all_traj["y_q_conf"].append(confs)
            for n, v in res_dict.items():
                if n in (track_and_plot+["y_q"]) and isinstance(v[0], torch.Tensor):
                    all_traj[n].append(torch.stack(v))  # n_steps, (n_query), attr_dim
            all_traj["query_accuracy"].append(task_acc)    # n_steps, 1
            # break in case if we have an infinite sequence, e.g. few-shot tasks
            if task_counter + 1 >= n_tasks:
                break
    
        if "ece" in track_and_plot:
            labels = torch.cat(all_traj["y_q"], dim=1)
            for confs, preds in zip(torch.cat(all_traj["y_q_conf"],dim=1),torch.cat(all_traj["y_q_pred"],dim=1)):  # n_steps,
                assert confs.dim() == preds.dim()
                all_traj["ece"].append(ece(confs=confs.flatten().cpu().tolist(),
                                           preds=preds.flatten().cpu().tolist(),
                                           labels=labels.flatten().cpu().tolist()))
            all_traj["ece"] = torch.tensor(all_traj["ece"])
        if  "query_accuracy" in track_and_plot:
            all_traj["query_accuracy"] = torch.cat(all_traj["query_accuracy"], dim=1)

        for attr in track_and_plot:
            traj = all_traj.get(attr)
            if traj is not None:
                all_traj[attr] = _get_mu_and_conf(traj, reduction_dim=1)  # Tuple(mu, std)
                plt_and_save_adaptation_trajectory(all_traj[attr],
                                                   path=os.path.join(self.args.logging.exp_dir,f"figures/{self.args.exp_identifier}/{dataset}"),
                                                   savename=f"{attr}.png")
        save_raw_pkl(all_traj,
                     path=os.path.join(self.args.logging.exp_dir,f"figures/{self.args.exp_identifier}/{dataset}"),
                     savename="raw_test_prediction.pkl")
        self.learner.train(model_train_state)
        return Metrics(*[all_traj[attr] for attr in track_and_plot]), best_step

    def find_best_finetune_configs(self, dataset, val_tasks, aug_func, finetune_mode):
        # we follow pmf and hps on 5 sampled tasks for the best lr for fine-tuning on each dataset
        if self.args.tasks == "md":
            inner_lrs = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
        else:
            inner_lrs = [0, 1e-7, 5e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
        num_finetune_steps = 50
        best_acc = 0
        best_step = -1 # use acc at the final step
        for lr in inner_lrs:
            trial_acc_traj = 0
            if self.args.alg !="smat":
                self.learner.lr = lr
                self.learner.num_iters = num_finetune_steps
            else:
                self.learner.args.meta_learner.inner_lr.lr = lr
                self.learner.args.meta_learner.num_finetune_steps = num_finetune_steps
            for task in tqdm(val_tasks):
                x_s,y_s,x_q,y_q = [_.to(self.device) for _ in task[:4]]
                source_label = y_q.new((1,)).fill_(-1)
                res_dict = self.learner(x_s=x_s,
                                        y_s=y_s,
                                        x_q=x_q,
                                        y_q=y_q,
                                        aug_func=aug_func,
                                        track_inner_trajectory=True,
                                        use_protomaml = self.args.use_protomaml,
                                        source_label = source_label,
                                        finetune_mode = finetune_mode)
                y_q_pred = torch.stack(res_dict["y_q_pred"]) # n_steps, n_query, nC
                trial_acc_traj += (y_q_pred.argmax(dim=-1)==y_q.view(1,-1)).float().mean(dim=-1) * 100/len(val_tasks)
           
            trial_best_acc_final = trial_acc_traj[-1].item()
            logger.info(f'{dataset:<30} LR: {lr} best final acc {trial_best_acc_final:.2f}')
            if trial_best_acc_final > best_acc:
                best_acc = trial_best_acc_final
                best_lr = lr
                best_step=-1
        # reset to best lr:
        if self.args.alg != "smat":
            self.learner.lr = best_lr
        else:
            self.learner.args.meta_learner.inner_lr.lr = best_lr
        logger.info(f'{dataset:<30} reset to best lr : {best_lr:5f} at best step : {best_step}')
        return best_step

    def test(self):
        test_dataset_results = {}
        test_loaders = self.datasetroutine.get_split_loaders(split="test",
                                                             args=self.args,
                                                             test_sources=self.args.test.datasets,
                                                             nTestEpisode = 600,
                                                             max_support_size_contrib_per_class = self.args.md_oneshot,
                                                             num_support = None if self.args.tasks =="md" else self.args.test.num_shots,
                                                             num_ways = None if self.args.tasks =="md" else self.args.test.num_ways,
                                                             return_source_info = self.args.train.source_info)
        val_loaders = {}
        print(f"testing on {list(self.args.test.datasets)}")

        for dataset, test_loader in test_loaders.items():
            v,best_step = self.evaluate_single_dataset(dataset=dataset,
                                                val_loader=val_loaders.get(dataset),
                                                test_loader=test_loader,
                                                n_tasks= self.args.test.max_tasks,
                                                track_and_plot=["query_accuracy", "ece"],
                                                finetune_mode=self.args.finetune_mode,
                                                hps=self.args.hps)

            logger.info(
                f"{dataset:<12} Acc: {v.query_accuracy[0][-1]:.2f} +/- {v.query_accuracy[1][-1]:.2f}, "
                f"Max: {v.query_accuracy[0].max():.2f}, "
                f"Final: {v.query_accuracy[0][best_step]:.2f}")
            logger.info(
                f"{dataset:<12} ECE : {v.ece[0][-1]:.4f} +/- {v.ece[1][-1]:.4f}, " 
                f"Final: {dataset:<12} ECE : {v.ece[0][best_step]:.4f} +/- {v.ece[1][best_step]:.4f}")
            test_dataset_results[dataset] = [v.query_accuracy[0][-1], v.query_accuracy[0].max(),v.query_accuracy[0][best_step]]  
        # display average results, returns in test_dataset_results are np arrays
        print("="*70)
        avg_all = np.stack([v for d, v in test_dataset_results.items()]).mean(0)
        logger.info(f"ALL: END: {avg_all[0]:.2f} MAX: {avg_all[1]:.2f}")
        print("="*70)

FP16 = False
if __name__ == "__main__":
    from src.utils import load_config_from_yaml, set_random_seed
    import argparse
    from rich.traceback import install

    install(show_locals=False)

    parser = argparse.ArgumentParser(description='run')
    parser.add_argument('--checkpoint', type=str, default="",help='test checkpoint path')
    parser.add_argument('--tasks', type=str, choices=["few-shot","md","md-nwaykshot"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hps", action="store_true",help= "whether to perform pmf-like hyper-parameter serach on 5 tasks")
    parser.add_argument("--aug", action="store_true",help= "whether to include data-augmentation during inner-loop adapatation")
    parser.add_argument("--alg", type=str, choices=["full","lora","pmf","smat"])
    parser.add_argument("--finetune_mode", type=str, default="", choices=["full","lora","pretrained+full","pretrained+lora"]) #TODO rename to make these less confusing
    parser.add_argument('--test_datasets', nargs='+', default="", help="overwrite the default testing datasets")
    parser.add_argument("--i",type=str)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    set_random_seed(seed=args.seed, deterministic=True)
    assert args.checkpoint
    configs = load_config_from_yaml(file_path="/"+os.path.join(*args.checkpoint.split("/")[:-2],
                                                           "exp_configs.yaml"),
                                    safe_load=False)
    # test configurations
    configs.devices = ["cuda:0","cuda:0"]
    configs.aug = args.aug
    configs.hps = args.hps
    configs.alg = args.alg
    configs.use_protomaml = False
    configs.exp_identifier = f"{args.alg}_{args.tasks}_hps{args.hps}_aug{args.aug}"
    configs.finetune_mode = args.finetune_mode
    configs.tasks = args.tasks

    if args.i:
        configs.exp_identifier += f"_{args.i}"


    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        filename=os.path.join(configs.logging.exp_dir, f"logs/{configs.exp_identifier}.log"),
                        filemode="a+")
    if args.n:
        configs.test.max_tasks = args.n


    configs.md_oneshot = None

    if "md" in args.tasks:
        if args.tasks=="md-nwaykshot":
            configs.test.num_ways = int(os.getenv("NWAYS", default=-1))
            configs.md_oneshot = configs.test.num_shots = int(os.getenv("KSHOTS", default=-1))
            configs.test.datasets = ['aircraft','omniglot','cu_birds','ilsvrc_2012','dtd','quickdraw','fungi','vgg_flower','traffic_sign','mscoco','cifar10','cifar100','mnist']
            logger.warning(f"{configs.test.num_ways}-ways-{configs.test.num_shots}-shots MD Subet Experiment")
        elif args.tasks=="md":
            configs.md_oneshot = 100 # = max no. support samples per class in MD
            configs.test.datasets = ['aircraft','omniglot','cu_birds','ilsvrc_2012','dtd','quickdraw','fungi','vgg_flower','traffic_sign','mscoco','cifar10','cifar100','mnist']
        assert configs.test.num_shots > 0
        if args.test_datasets:
            configs.test.datasets = args.test_datasets
        MetaDatasetDatamanager.name = "md"
        tester = MetaTester(configs,
                            DataManager=MetaDatasetDatamanager,
                            checkpoint = args.checkpoint,
                            )
        tester.test()

    elif args.tasks == "few-shot":
        configs.test.num_ways = int(os.getenv("NWAYS", default=-1))
        configs.test.num_shots = int(os.getenv("KSHOTS", default=-1))
        assert configs.test.num_ways > 1
        assert configs.test.num_shots > 0
        logger.warning(f"{configs.test.num_ways}-ways-{configs.test.num_shots}-shots Experiment")
        configs.test.datasets = ("pet","food","cars","sketch",)
        if args.test_datasets:
            configs.test.datasets = args.test_datasets
        FewShotDataManager.get_datalaoder = get_fewshot_taskloaders
        FewShotDataManager.name = "few-shot"
        tester = MetaTester(configs, DataManager=FewShotDataManager, checkpoint = args.checkpoint)
        tester.test()




