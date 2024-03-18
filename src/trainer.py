import math
from datetime import datetime
import torch
import os
import copy
from .learners import get_optimizer, get_scheduler, get_base_learner, get_meta_learner
from .taskdatasets import get_fewshot_taskloaders, get_metadataset_taskloaders
from .utils import Configuration, get_logger
import torch.nn as nn
from collections import namedtuple
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import trange, tqdm
from src.plots.ada_trajectory import plot_episode 
import logging
import random
import wandb



logger = get_logger(__name__)


class FewShotDataManager:
    pbar = None
    random_state = 0
    @staticmethod
    def set_pbar_description(message):
        FewShotDataManager.pbar.set_description(message)

    @staticmethod
    def tasks_in_epoch(task_loaders, devices, **kwargs):
        """500 tasks per domain in an epoch"""
        FewShotDataManager.pbar = trange(500, leave=True)
        if isinstance(task_loaders, dict):
            for _ in FewShotDataManager.pbar:
                datasets = list(enumerate(task_loaders.values()))
                random.Random(FewShotDataManager.random_state).shuffle(datasets)
                FewShotDataManager.random_state += 1
                for source_id, train_loader in datasets:
                    batch = next(train_loader)
                    for task in zip(*batch["train"], *batch["test"]):
                        yield [_.to(devices[0]) for _ in task] + [torch.tensor([source_id]).to(devices[0])] # x_s,y_s,x_q,y_q
        elif isinstance(task_loaders, (torch.utils.data.DataLoader,list)): 
            source_id = kwargs.get("source_id")
            assert source_id is not None
            for batch in task_loaders:
                for task in zip(*batch["train"], *batch["test"]):
                    yield [_.to(devices[0]) for _ in task] + [torch.tensor([source_id]).to(devices[0])] 

    @staticmethod
    def get_split_loaders(split, args, **kwargs):
        if split=="train":
            loader = get_fewshot_taskloaders(dataset_names=args.train.datasets,
                                             split="train",
                                             input_size=224,
                                             n_ways=args.train.num_ways,
                                             k_shots=args.train.num_shots,
                                             k_queries=args.train.num_queries,
                                             batch_size=1,
                                             aug=True)

            for n, p in loader.items():
                loader[n] = iter(p)
            return loader
        if split=="val":
             return get_fewshot_taskloaders(dataset_names=["painting","aircraft","real","quickdraw","sketch"],
                                            split="val",
                                            input_size=244,
                                            n_ways=args.test.num_ways,
                                            k_shots=args.train.num_shots,
                                            k_queries=args.train.num_queries,
                                            batch_size=1)
        if split=="test":
            return get_fewshot_taskloaders(dataset_names=args.test.datasets,
                                           split="test",
                                           input_size=224,
                                           n_ways=args.test.num_ways,
                                           k_shots=args.train.num_shots,
                                           k_queries=args.train.num_queries,
                                           batch_size=1)
        raise KeyError(f"undefined split {split} for few-shot dataset")


class MetaDatasetDatamanager:
    pbar = None

    @staticmethod
    def set_pbar_description(message):
        MetaDatasetDatamanager.pbar.set_description(message)

    @staticmethod
    def tasks_in_epoch(task_loaders, devices, **kwargs):
        """2000 tasks per domain in an epoch"""
        if isinstance(task_loaders, dict):
            MetaDatasetDatamanager.pbar = tqdm(task_loaders["single"], leave=True)  # train and test
        elif isinstance(task_loaders, torch.utils.data.DataLoader):  # evaluate and test
            MetaDatasetDatamanager.pbar = tqdm(task_loaders, leave=True)
        for task in MetaDatasetDatamanager.pbar:
            yield (*[_[0].to(devices[0]) for _ in task],)  # 1 task at a time

    @staticmethod
    def get_split_loaders(split, **kwargs):
        return get_metadataset_taskloaders(split=split, **kwargs)


class MetaTrainer:
    def __init__(self, args: Configuration, datasetroutine):
        self.args: Configuration = args
        self.devices = args.devices
        base_learner = get_base_learner(args)
        self.learner = get_meta_learner(base_learner, args)
        if self.devices[0] == self.devices[1]:
            self.learner.to(self.devices[0])
        self.opt = get_optimizer(self.learner.metatrain_parameters(), args.optimizer)
        self.scheduler = get_scheduler(self.opt, args.scheduler)
        ## using adam for hypernet
        second_optimizer = True
        if second_optimizer:
            self.opt_hypernet = torch.optim.AdamW(self.learner.mask_hypernet.get_trainable_selector_parameters(), lr=0.0001, weight_decay=1e-2)
            self.scheduler_hypernet = get_scheduler(self.opt_hypernet, args.scheduler)
        else:
            self.opt_hypernet = None
            self.scheduler_hypernet  = None

        self.loss_scaler = torch.cuda.amp.GradScaler() if self.args.train.fp16 else None
        self.datasetroutine = datasetroutine
        self.train_loader = datasetroutine.get_split_loaders(split="train",
                                                             args = args,
                                                             image_size = args.train.image_size,
                                                             base_sources = args.train.datasets,
                                                             return_source_info = args.train.source_info)
        self.val_loaders = datasetroutine.get_split_loaders(split="val",
                                                            args=args,
                                                            image_size = args.train.image_size,
                                                            nValEpisode = 120,
                                                            base_sources = args.train.datasets,
                                                            return_source_info = args.train.source_info)
        # # # use a fixed set of val_tasks
        if datasetroutine == FewShotDataManager:
            fixed_val_tasks = {}
            for n, v in self.val_loaders.items():
                fixed_val_tasks[n] = []
                for i, task in enumerate(tqdm(v, total=200)):
                    fixed_val_tasks[n].append(task)
                    if i + 1 >= 200:
                        break
                logger.info(f"using fixed validation tasks for {n}")
            self.val_loaders = fixed_val_tasks

    def _create_train_logger(self):
        timenow = datetime.now()
        log_dir = os.path.join(self.args.logging.exp_dir, "logs", timenow.strftime("run_%b_%m_%Y_%H_%M"))
        wandb.login()
        keywords =  self.args.logging.exp_dir.split("/")
        run_name = keywords[-1]
        tags = keywords[1:-1]
        wandb_logger = wandb.init(dir = ".",
                                  notes = os.getenv("notes"),
                                  config = self.args,
                                  project="SMAT",
                                  name=run_name,
                                  tags=tags)
        return wandb_logger

    def save_checkpoint(self, model, optimizer, scheduler, loss_scaler, path):
        save_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scheduler:
            save_state.update({"scheduler": self.scheduler.state_dict()})
        if loss_scaler:
            save_state.update({"loss_scaler": loss_scaler.state_dict()})
        torch.save(save_state, path)

    def load_checkpoint(self,path):
        assert os.path.isfile(path), path
        checkpoint = torch.load(path)
        self.learner.load_state_dict(checkpoint["model"])
        print(f"succesfully loaded MODEL state_dict from {path}")             
        try:
            if checkpoint.get("optimizer"):
                self.opt.load_state_dict(checkpoint["optimizer"])
            else:
                raise KeyError
        except KeyError:
            logger.warning(f"{path} does not have OPTIMIZER state_dict")
        else:
            print(f"succesfully loaded OPTIMIZER state_dict from {path}")

        try:
            if checkpoint.get("scheduler"):
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            else:
                raise KeyError
        except KeyError:
            logger.warning(f"{path} does not have SCHEDULER state_dict")
        else:
            print(f"succesfully loaded SCHEDULER state_dict from {path}")

        try:
            if checkpoint.get("loss_scaler"):
                self.loss_scaler.load_state_dict(checkpoint["loss_scaler"])
            else:
                raise KeyError
        except KeyError:
            logger.warning(f"{path} does not have LOSS_SCALER state_dict")
        else:
            print(f"succesfully loaded LOSS_SCALER state_dict from {path}")

    def evaluate_accuracy(self, dataset, evaluate_loader, n_tasks, **kwargs)->namedtuple:
        def _get_mu_and_conf(traj)->(np.ndarray, np.ndarray):
            """
            Parameters
            ----------
            traj : n_tasks, n_steps

            Returns
            -------

            """
            if traj.dim()==1:
                traj_mu = traj.cpu().numpy()
                traj_conf = np.zeros_like(traj_mu)
            elif traj.dim()==2:
                traj_mu = traj.mean(0).cpu().numpy()
                traj_conf = ((1.96 * traj.std(0)) / math.sqrt(traj.size(0))).cpu().numpy()
            else:
                raise Exception(traj.shape)
            return traj_mu, traj_conf

        model_train_state = self.learner.training
        self.learner.eval()
        Metrics = namedtuple('Metrics', "query_accuracy")
        accuracy_traj = []
        for task_counter, task in enumerate(self.datasetroutine.tasks_in_epoch(evaluate_loader, devices=self.devices, **kwargs)):
            x_s, y_s, x_q, y_q = task[:4]
            source_label = task[-1] if self.args.train.source_info else y_q.new((1,)).fill_(-1)
            with torch.cuda.amp.autocast(self.args.train.fp16):
                res_dict = self.learner(x_s, y_s, x_q, y_q=None, source_label = source_label)
            accuracy_traj.append((torch.stack(res_dict["y_q_pred"]).to(self.devices[1]).argmax(dim=-1)==y_q.to(self.devices[1]).view(1,-1)).float().mean(dim=-1)) # n_steps
            self.datasetroutine.set_pbar_description(f'{dataset:<12} : sparsity : {res_dict["train/sparsity"][-1].item():.3f} accuracy {accuracy_traj[-1][-1].item():.3f}')
            # break in case if we have an infinite sequence
            if task_counter+1>=n_tasks:
                break
        # stack in the task dimension,
        accuracy_traj = torch.stack(accuracy_traj) * 100 # n_tasks, n_steps
        accuracy_traj_np = _get_mu_and_conf(accuracy_traj)  # Tuple(mu, std)
        self.learner.train(model_train_state)
        return Metrics(accuracy_traj_np)

    def _sanity_val_run(self, n_tasks):
        logger.info(f"{f'Evaluating pretrained model...': ^40}")
        pretrained_results = {}
        for source_id, (dataset, val_loader) in enumerate(self.val_loaders.items()):
            named_tup = self.evaluate_accuracy(dataset, val_loader, n_tasks=n_tasks, source_id = source_id)
            pretrained_results[dataset] = (named_tup.query_accuracy[0], named_tup.query_accuracy[1])
            n =  dataset
            v = named_tup.query_accuracy[0], named_tup.query_accuracy[1]
            print(f"{n:<12} : {v[0][-1]:.2f} +/- {v[1][-1]:.2f}")
        return np.array([v[0][-1] for n,v in pretrained_results.items() if n in self.args.train.datasets])

    def backward_propagation(self, loss_dict, param_list, retain_grad):
        loss_scaler = lambda x: self.loss_scaler.scale(x) if self.args.train.fp16 else x
        num_loss_terms = len(loss_dict)
        if not self.args.train.gradient_surgery:
            sum(loss_scaler(loss_dict.values())).backward(retain_graph=retain_grad)
        else:
            raise NotImplementedError

    def meta_train(self, resume_exp_dir, resume_ckpt):
        global_step = 0   # counts the number of training task batches
        tb_logger = self._create_train_logger()
        patience = self.args.train.max_patience
        best_val_acc = np.zeros((len(self.val_loaders)))
        print(f"Training for {self.args.train.max_epoch} epochs")
        if resume_exp_dir:
            self.load_checkpoint(os.path.join(resume_exp_dir,"checkpoints","{}.pt".format(resume_ckpt)))
            try:
                resume_epoch = int(resume_ckpt.split("_")[-1])
            except ValueError:
                resume_epoch = 0
            logger.info(f"resume training from epoch {resume_epoch}")
        else:
            resume_epoch = 0
        _ = self._sanity_val_run(n_tasks=1)
        self.opt.zero_grad()
        if self.opt_hypernet:
            self.opt_hypernet.zero_grad()

        # training loop
        for epoch in range(resume_epoch, self.args.train.max_epoch):
            self.learner.train()
            # record learning rates for the current epoch
            for i, param_groups in enumerate(self.opt.param_groups):
                if not epoch:
                    self.opt.param_groups[i]['lr'] = 1e-6  # minimum learning rate for the start of training
                logger.info(f"updated learning rate for param {self.opt.param_groups[i]['name']} "
                            f"to {self.opt.param_groups[i]['lr']:.7f}")      
            # train for one epoch
            for task_counter, task in enumerate(
                    self.datasetroutine.tasks_in_epoch(self.train_loader, devices=self.devices)):
                x_s, y_s, x_q, y_q = task[:4]
                source_label = task[-1] if self.args.train.source_info else y_q.new((1,)).fill_(-1)
                # plot_episode(x_s,y_s,x_q,y_q, filename=f"{source_label.item()}_task_{task_counter}")
                with torch.cuda.amp.autocast(self.args.train.fp16):
                    res_dict = self.learner(x_s, y_s, x_q, y_q, source_label=source_label)
                # bp for a single task
                for loss_item, loss_value in res_dict["outer_loss"].items():
                    if not torch.isfinite(loss_value):
                        logger.error(f"training becomes unstable at epoch : {epoch} for loss {loss_item} : {loss_value}")
                        return best_val_acc, epoch + 1
                    loss_value /= self.args.train.batchsize
                # backward
                self.backward_propagation(res_dict["outer_loss"],
                                          self.learner.metatrain_parameters(as_list=True), 
                                          retain_grad = False #bool((task_counter+1) % self.args.train.batchsize)
                                          )
                # reformat the learner;s loss for logging
                outer_loss_dict = res_dict.pop("outer_loss")
                for loss_name, loss_value in outer_loss_dict.items():
                    res_dict[f"outer_loss/{loss_name}"] = loss_value.item()
                # add trainer's logs for logging
                res_dict.update({f"learning_rates/param_group_{i}": self.opt.param_groups[i]["lr"] for i in range(len(self.opt.param_groups))})
                # display progress bar
                log_msg = res_dict.pop("log_msg")
                self.datasetroutine.set_pbar_description(log_msg)
                # meta update, i.e. optimizer.step()
                if (task_counter+1) % self.args.train.batchsize == 0:
                    if self.args.train.fp16:
                        if self.args.optimizer.max_grad_norm > 0:
                            self.loss_scaler.unscale_(self.opt)
                            nn.utils.clip_grad_norm_(self.learner.clip_grad_parameters(), self.args.optimizer.max_grad_norm)
                            if self.opt_hypernet:
                                self.loss_scaler.unscale_(self.opt_hypernet)
                            nn.utils.clip_grad_norm_(self.learner.mask_hypernet.get_trainable_selector_parameters(), 1.)
                        self.loss_scaler.step(self.opt)
                        if self.opt_hypernet:
                            self.loss_scaler.step(self.opt_hypernet)
                        self.loss_scaler.update()
                    else:
                        if self.args.optimizer.max_grad_norm > 0:
                            nn.utils.clip_grad_norm_(self.learner.clip_grad_parameters(), self.args.optimizer.max_grad_norm)
                        self.opt.step()
                    # clear grads
                    self.opt.zero_grad()
                    if self.opt_hypernet:
                        self.opt_hypernet.zero_grad()
                    # log training stats for the last task in a single batch
                    tb_logger.log(res_dict, step=global_step)
                    global_step += 1

            # validation after N epoch
            if (epoch + 1) % self.args.logging.val_freq == 0:
                val_results= {}
                for source_id, (dataset, val_loader) in enumerate(self.val_loaders.items()):
                    val_results[dataset] = self.evaluate_accuracy(dataset, val_loader, n_tasks=120, source_id=source_id)
                logger.info(f"{f'Validation Acc@{epoch + 1}': ^40}")
                for n,v in val_results.items():
                    logger.info(f"{n:<12} : {v.query_accuracy[0][-1]:.2f} +/- {v.query_accuracy[1][-1]:.2f}")
                md_val_datasets = ('aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower')
                val_final_avg_acc = np.array([v.query_accuracy[0][-1] for d,v in val_results.items() if d in md_val_datasets])
                tb_logger.log({f"train/val_accuracy/{n}":v.query_accuracy[0][-1] for n,v in val_results.items()}, global_step)
                tb_logger.log({f"train/val_accuracy/average": val_final_avg_acc.mean().item()}, global_step)
                logger.info(f"IID Validation Avg: {val_final_avg_acc.mean().item():.2f} Best: {best_val_acc.mean().item():.2f} ")
                if val_final_avg_acc.mean() > best_val_acc.mean():  # better on average
                    patience = self.args.train.max_patience
                    best_val_acc = val_final_avg_acc
                    best_val_state_dict = copy.deepcopy(self.learner.state_dict())
                    self.save_checkpoint(
                        model = self.learner,
                        optimizer= self.opt,
                        scheduler= self.scheduler,
                        loss_scaler = self.loss_scaler,
                        path=os.path.join(self.args.logging.exp_dir, "checkpoints", f"best_validation.pt"))
                    logger.info("Best validation model is updated")

                else:
                    patience-= 1
                if patience <= 0:
                    print("early stooping")
                    break

            if self.scheduler:
                self.scheduler.step(epoch+1)
                if self.scheduler_hypernet:
                    self.scheduler_hypernet.step(epoch+1)
            else:
                if (epoch+1) % 5 == 0:
                    for i, param_groups in enumerate(self.opt.param_groups):
                        if param_groups["name"] in ["delta_params"]:
                            self.opt.param_groups[i]['lr'] = max(self.opt.param_groups[i]['lr']*0.5, 0.00005)
                            
            # save every N interations
            if (epoch + 1) % self.args.logging.save_freq == 0:
                self.save_checkpoint(
                    model=self.learner,
                    optimizer=self.opt,
                    scheduler=self.scheduler,
                    loss_scaler=self.loss_scaler,
                    path= os.path.join(self.args.logging.exp_dir, "checkpoints", f"iter_{epoch + 1}.pt"))
                
        # end of training
        self.learner.load_state_dict(best_val_state_dict)
        return best_val_acc, epoch+1



if __name__ =="__main__":
    from .utils import load_config_from_yaml,save_config_to_yaml,ensure_path, set_random_seed, save_all_py_files_in_src
    import argparse
    from rich.traceback import install

    install(show_locals=False)

    parser = argparse.ArgumentParser(description='run')
    parser.add_argument('--mode', nargs='+', default=["train"], choices=["train"],
                        help="mode")
    parser.add_argument('--checkpoint', type=str, default="",
                        help='a path to a .pt/.pth checkpoint for testing/evaluation')
    parser.add_argument("--yaml", type=str, default="",
                        help="a path to .yaml file for loading training/testing configurations")
    parser.add_argument("--resume_exp_dir", type=str, default="",
                        help="the experiment directory for resuming training")
    parser.add_argument("--resume_ckpt", type=str, default="",
                        help="the epoch to resume training from")
    parser.add_argument("--seed", type=int, default=0, help="global random seed for the run")
    parser.add_argument("--wandb_offline", action="store_true", help="wandb offline mode")
    parser.add_argument("--wandb_disable", action="store_true", help="disable wandb")
    parser.add_argument("--mp", action="store_true", help="whether to use model parallelism across two GPUs")
    parser.add_argument("--debug", action="store_true", help="whether to log and display debug-level messages")
    args = parser.parse_args()

    set_random_seed(seed=args.seed, deterministic=True)

    os.environ['WANDB_MODE'] = 'online'
    if args.wandb_disable:
        os.environ['WANDB_MODE'] = 'disabled'
    elif args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    if "train" in args.mode:
        if args.resume_exp_dir:
            configs = load_config_from_yaml(file_path=os.path.join(args.resume_exp_dir,"exp_configs.yaml"), safe_load=False)
            prefix = configs.logging.exp_dir
            logfile_mode = "a"
        else:
            configs = load_config_from_yaml(file_path=f"configs/{args.yaml}.yaml")
            prefix = configs.logging.exp_dir
            ensure_path(configs.logging.exp_dir)
            ensure_path(os.path.join(configs.logging.exp_dir, "logs"))
            ensure_path(os.path.join(configs.logging.exp_dir, "checkpoints"))
            save_config_to_yaml(configs, configs.logging.exp_dir + '/exp_configs.yaml')
            save_all_py_files_in_src(source_path='src', destination_path=configs.logging.exp_dir + '/code_timestamp')
            logfile_mode = "w"

        if args.mp:
            configs.devices = ["cuda:0","cuda:1"]
        else:
            configs.devices = ["cuda:0","cuda:0"]

        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG if args.debug else logging.INFO,
                            filename=os.path.join(configs.logging.exp_dir,"logs/train.log"),
                            filemode=logfile_mode)

        trainer = MetaTrainer(configs, MetaDatasetDatamanager)
        trainer.meta_train(args.resume_exp_dir, args.resume_ckpt)







