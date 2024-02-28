import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Generator
from src.utils import Configuration, accuracy, get_logger
import math
from torch.func import functional_call
from collections import defaultdict, OrderedDict
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from functools import partial
from copy import deepcopy

logger = get_logger(__name__)


class DeltaParams(nn.Module):
    def __init__(self, meta_params):
        super(DeltaParams, self).__init__()
        self.phis = torch.nn.ParameterList()
        self.param_name_2_phi = OrderedDict()
        for i, (n,p) in enumerate(meta_params):
            self.phis.append(torch.nn.Parameter(p.detach().clone(),requires_grad=True))
            self.param_name_2_phi[n] = i
        for p in self.phis:
            nn.init.zeros_(p)
        p_count = 0
        for p in self.parameters():
            p_count += p.numel()
        logger.info(f"Total Number of Parameters in the delta_param hypernetwork :{p_count}")

    def get_fast_params(self):
        return {n: p.clone().requires_grad_(True) for n, p in self.named_parameters()}

    def forward(self,phis:dict):
        """

        Parameters
        ----------
        phis : dict ( phis_param_names, phis)

        Returns
        -------
        delta_params : dict (net_param_names, delta_params)

        """
        delta_params={}
        for net_param_name, phi_idx in self.param_name_2_phi.items():
            delta_params[net_param_name] = phis[f"phis.{phi_idx}"].clone()
        return delta_params

    def set_fast_phi_for_net_param(self, fast_phis, target_names_values):
        for net_param_name, new_value in target_names_values.items():
            if self.param_name_2_phi.get(net_param_name):
                phi_idx = self.param_name_2_phi[net_param_name]
                fast_phis[f"phis.{phi_idx}"] = new_value
            else:
                raise KeyError(f"the base learner backbone does not contain parameter : {net_param_name}")

    def _get_deltaparam_attribute_for_net_param(self, net_param_name):
        """debug function, not used in training -> delta_param.shape, hypernet_paramnae"""
        phi_idx = self.param_name_2_phi.get(net_param_name)
        if phi_idx is None:
            return "","n/a"
        else:
            return f"phis.{phi_idx}", np.array2string(torch.tensor(self.phis[phi_idx].shape).cpu().numpy())


class MasksCollection(nn.Module):
    def __init__(self, meta_params, num_masks, hid_dim, init_dropout_rate=0.0, structured_sparsity=False, bias_sparsity=True, stochastic_mask=True, fix_embeddigs=False, spectral_norm=True):
        super(MasksCollection, self).__init__()
        from src.learners.base_learners.torchhub_vit import Block
        self.hid_dim = hid_dim
        self.key_dim = hid_dim
        self.num_masks = num_masks
        self._initalize_masks(num_masks,
                              meta_params=meta_params,
                              init_dropout_rate=init_dropout_rate,
                              stochastic_mask=stochastic_mask,
                              structured_sparsity=structured_sparsity,
                              bias_sparsity=bias_sparsity,
                              fix_embeddigs=fix_embeddigs)

        self.masks_keys = nn.Parameter(nn.init.kaiming_normal_(torch.FloatTensor(num_masks, self.key_dim)), requires_grad=True)
        self.count_encoding = nn.Parameter(nn.init.kaiming_normal_(torch.FloatTensor(1, 1, self.key_dim)), requires_grad=True)
        self.input_embed_proj = nn.Linear(384, hid_dim)
        self.skills_tokens = nn.Parameter(nn.init.normal_(torch.FloatTensor(1, 4, self.key_dim), std=1e-3), requires_grad=True)
        self.task_embed_agg = Block(dim=hid_dim,
                                    num_heads=4,
                                    mlp_ratio=2,
                                    )

        if spectral_norm:
            sn = nn.utils.spectral_norm

            def _add_sn(m):
                if isinstance(m, nn.Linear):
                    logger.info("added spectral norm !")
                    return sn(m)
                else:
                    return m

            self.apply(_add_sn)

        self.epsilon = 1e-6
        self.register_buffer("zeta", torch.tensor(1.1))
        self.register_buffer("gamma", torch.tensor(-0.1))
        self.sampled_masks_collection = None
        ## count the number of trainable parameters
        p_count = 0
        for p in self.parameters():
            p_count += p.numel()
        logger.info(f"Total Number of Parameters in the masks and hypernetwork :{p_count}")

    def _initalize_masks(self, num_masks, meta_params, init_dropout_rate=0.0, structured_sparsity=False, bias_sparsity=True, stochastic_mask=True, fix_embeddigs=False) -> None:
        self.log_alphas = torch.nn.ParameterList()
        self.log_alpha_2_scale = OrderedDict()
        self.param_name_2_log_alpha_scale = OrderedDict()  # dict of log_alphas_name : (mask_name, dim_scale)
        log_alpha_init = math.log(1 - init_dropout_rate) - math.log(init_dropout_rate)
        log_alpha_idx = 0
        log_alpha_init = 0.0
        log_alpha_init_std = 0.5
        for param_n, p in meta_params:
            if structured_sparsity:
                expand_to_shape = (*p.shape, num_masks)
                if fix_embeddigs and ".embeddings." in param_n:
                    log_alpha = torch.nn.Parameter(p.new(*p.shape, num_masks).normal_(log_alpha_init, log_alpha_init_std), requires_grad=True)
                    dim_scale = 1.0
                elif ".weight" in param_n:
                    if p.dim() == 1:
                        log_alpha = torch.nn.Parameter(p.new(*p.shape, num_masks).normal_(log_alpha_init, log_alpha_init_std), requires_grad=True)
                        dim_scale = 1.0
                    elif p.dim() == 2:  # output, input
                        log_alpha = torch.nn.Parameter(torch.Tensor(1, p.size(1), num_masks).normal_(log_alpha_init, log_alpha_init_std), requires_grad=True)
                        dim_scale = p.size(0)
                    elif p.dim() == 4:  # conv
                        log_alpha = torch.nn.Parameter(torch.Tensor(p.size(0), 1, 1, 1, num_masks).normal_(log_alpha_init, log_alpha_init_std), requires_grad=True)
                        dim_scale = p.numel() / p.size(0)
                        # bias_with_tied_masks.append((param_n.replace("weight", "bias"),log_alpha_idx))
                    else:
                        raise NotImplementedError(f"unhandled weight param {param_n} of dim : {p.shape}")
                elif ".bias" in param_n:
                    if bias_sparsity:
                        log_alpha = torch.nn.Parameter(p.new(*p.shape, num_masks).normal_(log_alpha_init, log_alpha_init_std), requires_grad=True)
                        dim_scale = 1.0
                    else:
                        log_alpha = None
                        dim_scale = p.numel()
                else:
                    log_alpha = torch.nn.Parameter(p.new(*p.shape, num_masks).normal_(log_alpha_init, log_alpha_init_std), requires_grad=True)
                    dim_scale = 1.0
                    logger.warning(f"un-handled parameter group type for : {param_n}, default mask.shape == param.shape")
            else:
                expand_to_shape = None
                log_alpha = torch.nn.Parameter(p.new(*p.shape, num_masks).normal_(log_alpha_init, log_alpha_init_std), requires_grad=True)
                dim_scale = 1.0

            if log_alpha is None:
                self.param_name_2_log_alpha_scale[param_n] = (None, dim_scale, expand_to_shape)  # None mask is default to always-on
            else:
                self.log_alphas.append(log_alpha)
                self.log_alpha_2_scale[f"log_alphas.{log_alpha_idx}"] = dim_scale
                self.param_name_2_log_alpha_scale[param_n] = (log_alpha_idx, dim_scale, expand_to_shape)
                log_alpha_idx += 1

        self.register_buffer("beta", torch.tensor(2 / 3, requires_grad=False))
        self.register_buffer("zeta", torch.tensor(1.1, requires_grad=False))
        self.register_buffer("gamma", torch.tensor(-0.1, requires_grad=False))
        self.register_buffer("use_hard_masks", torch.tensor(0, requires_grad=False))
        self.epsilon = 1e-6
        self.stochastic_mask = stochastic_mask
        self.fix_embeddings = fix_embeddigs
        # check everything is okay
        with torch.no_grad():
            fast_log_alphas = self.get_fast_params()
            mask, sparsity = self._sample_all_masks(fast_log_alphas)
            expected_z = 0
            for i, v in enumerate(mask.values(), 1):
                expected_z += v.flatten(end_dim=-2).mean(dim=0)
            sp_loss = self.sparse_penalty(log_alphas=fast_log_alphas)
            for m, masks_infos in enumerate(zip(sparsity, expected_z, sp_loss)):
                sp, z, spl = masks_infos
                logger.info(f"Variational Masks-{m} initialized with Sparisty : {sp.item():.3f}")
                logger.info(f"Variational Masks-{m} initialized with Expected Median : {z / i:.3f}")
                logger.info(f"Variational Masks-{m} initialized with Sparisty loss : {spl.item():.5f}")

    def _sample_all_masks(self, log_alphas: dict, **kwargs) -> Tuple[dict, torch.Tensor]:
        """
        Parameters
        ----------
        log_alphas : dict ( log_alpha_param_names, log_alpha)

        Returns
        -------
        masks : dict ( net_param_names, mask)
        """
        on_aprior = ["head"]  # the head is always on
        masks = {}
        non_zero, total = torch.tensor([0.0] * self.num_masks, device=self.beta.device), 0.0

        for net_param_name, idx_scale_reshape in self.param_name_2_log_alpha_scale.items():
            log_alpha_idx, dim_scale, expand_to_shape = idx_scale_reshape
            if self.fix_embeddings and ".embeddings." in net_param_name:
                log_alpha = log_alphas[f"log_alphas.{log_alpha_idx}"]
                masks[net_param_name] = torch.zeros_like(log_alpha)
                non_zero += 0  # embeddings is off, do nothing
                total += log_alpha.numel() * dim_scale
            elif any(k in net_param_name for k in on_aprior):  # always keep the head on, otherwise training is not stable
                log_alpha = log_alphas[f"log_alphas.{log_alpha_idx}"]
                masks[net_param_name] = torch.ones_like(log_alpha)
                non_zero += log_alpha.numel() * dim_scale
                total += log_alpha.numel() * dim_scale
            elif log_alpha_idx is None:
                masks[net_param_name] = torch.ones(1, device=self.zeta.device)
                non_zero += dim_scale / self.num_masks
                total += dim_scale
            else: # sample a mask from the variational distribution
                log_alpha = log_alphas[f"log_alphas.{log_alpha_idx}"]
                if self.training and self.stochastic_mask:
                    # sample a mask
                    u = torch.empty(log_alpha.shape, device=log_alpha.device).uniform_(0.5 - self.epsilon, 0.5 + self.epsilon)
                    log_u = torch.log(u) - torch.log(1 - u)
                    beta = self.beta
                else:
                    # for inference
                    log_u = 0.0
                    beta = 1.0  
                s = F.sigmoid((log_u + log_alpha) / beta)
                s_bar = s * (self.zeta - self.gamma) + self.gamma
                z = F.hardtanh(s_bar, min_val=0, max_val=1)
                non_zero += torch.greater(z, torch.tensor([0.], dtype=z.dtype, device=z.device)).view(-1, self.num_masks).sum(dim=0) * dim_scale
                total += log_alpha.numel() * dim_scale
                # if self.hard_mask:
                #     z = torch.greater(z, 0).float() # convert to {0,1}
                if expand_to_shape:
                    z = z.expand(*expand_to_shape)
                masks[net_param_name] = z
        total /= self.num_masks
        masks_sparsities = 1. - non_zero / total
        return masks, masks_sparsities

    def cdf_qz(self, tau, log_alpha):
        xn = (tau - self.gamma) / (self.zeta - self.gamma)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.beta - log_alpha).clamp(min=self.epsilon, max=1 - self.epsilon)

    def sparse_penalty(self, log_alphas: dict):
        expcted_l0 = torch.tensor([0.0] * self.num_masks, device=self.beta.device)
        normalizer = 0  # normalize by the number of model parameters
        for log_alpha_name, log_alpha in log_alphas.items():
            # 1 - cdf(0)
            dim_scale = self.log_alpha_2_scale[log_alpha_name]
            mins_cdf = 1.0 - self.cdf_qz(tau=0, log_alpha=log_alpha)
            expcted_l0 += (mins_cdf * dim_scale).flatten(end_dim=-2).sum(dim=0)
            normalizer += log_alpha.numel() * dim_scale / self.num_masks
        return expcted_l0 / normalizer  # this makes penalty always within [0,1]

    def lp_penalty(self, masks, named_params, structured=False, p=1):
        raise NotImplementedError

    def clip_log_alpha(self):
        for log_alpha in self.log_alphas:
            log_alpha.data.clamp_(min=math.log(0.01), max=math.log(100))

    def get_fast_params(self, **kwargs):
        return {n: p.clone().requires_grad_(True) for n, p in self.named_parameters() if "log_alphas" in n}

    def _get_mask_attribute_for_net_param(self, net_param_name):
        """debug function, not used in training -> mask.shape, dim_scale"""
        mask_attr = self.param_name_2_log_alpha_scale.get(net_param_name)
        if mask_attr:
            log_alpha_idx, dim_scale, mask_reshape = mask_attr
            if log_alpha_idx is not None:
                log_alpha = self.log_alphas[log_alpha_idx]
                if mask_reshape:
                    log_alpha = log_alpha.reshape(mask_reshape)
                return f"log_alphas.{log_alpha_idx}", np.array2string(torch.tensor(log_alpha.shape).cpu().numpy()), dim_scale
            else:
                return "", "[1]", dim_scale
        else:
            return "", "n/a", "n/a"

    def get_task_representation(self, probe_model, x_s, y_s, x_q):
        # get task mean, protoype, and prototype variance
        with torch.no_grad():
            n_classes = y_s.max() + 1
            x_s_f = probe_model(x_s, True, )  # input args are (x_s, return_features_only)
            tokens, counts = [], []
            y_s_1hot = F.one_hot(y_s, n_classes).T  # nC, nSupp
            prototypes = torch.mm(y_s_1hot.float(), x_s_f)  # nC, feat_dim
            prototypes = prototypes / y_s_1hot.sum(dim=1, keepdim=True)
        tokens = torch.cat([prototypes.unsqueeze(0), self.skills_tokens], dim=1)
        assert torch.isfinite(tokens.mean()), tokens.mean()
        task_embed = self.task_embed_agg(tokens, return_attention=False)
        return task_embed[0:1, 0, :]

    def sample_scores_from_logits(self, logits, temperature, sample=True, stround=True):
        if self.training and sample:
            scores = s = RelaxedBernoulli(temperature=temperature, logits=logits).rsample()
        else:
            scores = s = F.sigmoid(logits / temperature)
        if stround:
            scores = torch.round(s) - s.detach() + s
        return s, scores

    def get_masks_selection_scores(self, task_embed, temperature):
        dot_atta = torch.mm(self.masks_keys, task_embed.T).squeeze(-1)
        s, scores = self.sample_scores_from_logits(logits=dot_atta, temperature=temperature, sample=True, stround=False)
        return dot_atta, s, scores

    def get_seperate_masks(self):        
        with torch.no_grad():
            log_alphas = self.get_fast_params()
            all_masks, _ = self._sample_all_masks(log_alphas=log_alphas)
            seperate_masks = []
            for mask_idx in range(self.num_masks):
                seperate_masks.append([{n: m.unbind(dim=-1)[mask_idx] for n,m in all_masks.items()}])
        return seperate_masks

    def merge_masks_with_scores(self, log_alphas_collections, scores):
        # merge the masks by addtion, now
        assert len(scores) == self.num_masks, scores.shape
        merged_masks = {}
        non_zero, total, merged_flag = 0.0, 0.0, False
        union_loss = torch.tensor([0.], device=scores.device)
        # sample 2 index
        if self.num_masks > 1:
            union_loss_prob = torch.nan_to_num(F.relu(scores.detach())) + 1e-4
            union_loss_idx = torch.multinomial(union_loss_prob, replacement=False, num_samples=2)
            union_loss_scores = F.one_hot(union_loss_idx, num_classes=self.num_masks).sum(dim=0)
        else:
            union_loss_scores = torch.zeros_like(scores,requires_grad=False)
        # mask_overlap
        sampled_masks, mask_sparsities = self._sample_all_masks(log_alphas=log_alphas_collections)
        for param_name, m in sampled_masks.items():
            if "head" in param_name:
                param_mask = torch.ones_like(m, requires_grad=False).mean(dim=-1) 
            else:
                param_mask = torch.mul(m, scores).sum(dim=-1)  # weighted sum of all masks
            union_loss += torch.mul(m, union_loss_scores).sum()
            total += param_mask.numel()
            non_zero += torch.greater(param_mask, 0).float().sum()  
            merged_masks[param_name] = param_mask
        merged_sparsity = 1. - non_zero / total
        union_loss /= total
        return merged_masks, merged_sparsity, mask_sparsities, union_loss

    def forward(self, probe_model, x_s, y_s, x_q, log_alphas_collections, temperature):
        task_embed = self.get_task_representation(probe_model, x_s, y_s, x_q) 
        dot_attn_scores, soft_selection_scores, selection_scores = self.get_masks_selection_scores(task_embed, temperature=temperature)
        if self.num_masks > 1: 
            selection_scores = selection_scores / (selection_scores.sum() + 1e-8)
        merged_masks, merged_masks_sparsity, sep_mask_sparsities, union_loss = self.merge_masks_with_scores(log_alphas_collections, selection_scores)
        return task_embed, merged_masks, merged_masks_sparsity, sep_mask_sparsities, soft_selection_scores, selection_scores, dot_attn_scores, union_loss

    def clip_log_alpha(self):
        for log_alpha in self.get_trainable_mask_parameters():
            log_alpha.data.clamp_(min=math.log(0.01), max=math.log(100))

    def get_trainable_mask_parameters(self):
        for n, p in self.named_parameters():
            if "log_alphas" in n:
                yield p

    def get_trainable_selector_parameters(self):
        for n, p in self.named_parameters():
            if "log_alphas" not in n:
                yield p


class SparseInterpolatedExperts(nn.Module):
    def __init__(self, net: nn.Module, args: Configuration):
        super().__init__()
        self.args = args
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.nonparametric_head = net.nonparametric_head
        self.inner_param_names = {n: 1 for n, _ in net.inner_meta_params()}
        # delta_params wil alwways include all trainable parameters
        self.delta_hypernet = DeltaParams(meta_params=net.outer_meta_params())
        self.num_masks = args.meta_learner.sparsity.num_experts
        self.hid_dim = 384
        self.mask_hypernet = MasksCollection(meta_params=list(net.outer_meta_params()),
                                             num_masks=self.num_masks,
                                             hid_dim=self.hid_dim,
                                             init_dropout_rate=0.05,
                                             structured_sparsity=args.meta_learner.sparsity.structured,
                                             bias_sparsity=args.meta_learner.sparsity.on_bias,
                                             stochastic_mask=args.meta_learner.sparsity.stochastic_mask,
                                             fix_embeddigs=args.meta_learner.sparsity.fix_embeddings)
        # learnable lrs for delta_params and grad_mask_params
        self.lrs = args.meta_learner.inner_lr.lr
        self.lagrangian_multiplier = nn.Parameter(torch.FloatTensor(self.num_masks).fill_(0), requires_grad=True)
        self.register_buffer("mask_sparsity_targets", torch.tensor([args.meta_learner.sparsity.target] * self.num_masks))
        self.register_buffer("num_metaupdates", torch.tensor([0.0]))

    def _interpolate_params(self, masks: dict, delta_params: dict, last_iter: bool = False) -> dict:
        combined_weights = {}
        for net_p_name, delta_p in delta_params.items():
            if "sp_mod" in self.args.meta_learner.sparsity.sparsify_where:
                combined_weights[net_p_name] = masks[net_p_name] * delta_p + self.net.get_parameter(net_p_name)
            elif "mod" in self.args.meta_learner.sparsity.sparsify_where:
                combined_weights[net_p_name] = delta_p + self.net.get_parameter(net_p_name)
            else:
                raise NotImplementedError(f"{self.args.meta_learner.sparsity.sparsify_where} contains undefined sparsifying method")
        return combined_weights

    def _config_head(self, x_s, y_s, fast_weight_init) -> dict:
        n_classes = torch.max(y_s) + 1
        fast_names_values = {}
        reattach_names_values = {}
        if self.args.meta_learner.zero_trick:
            # use 0 init for weights and bias
            weight = torch.zeros(n_classes, self.net.feat_dim, device=x_s.device, requires_grad=True)
            fast_names_values["head.weight"] = weight
            if self.args.base_learner.bias:
                bias = torch.zeros(n_classes, device=x_s.device, requires_grad=True)
                fast_names_values["head.bias"] = bias
        else:
            # use protomaml initalization for weights and bias, ( first order only)
            x_s_f = functional_call(self.net, fast_weight_init, (x_s, True,))  # input args are (x_s, return_features_only)
            y_s_1hot = F.one_hot(y_s, n_classes).T  # nC, nSupp
            prototypes = torch.mm(y_s_1hot.float(), x_s_f)  # nC, feat_dim
            prototypes = prototypes / y_s_1hot.sum(dim=1, keepdim=True)
            weight = prototypes #2 * prototypes
            reattach_names_values["head.weight"] = weight
            fast_names_values["head.weight"] = weight.detach().requires_grad_(True)
            if self.args.base_learner.bias:
                bias = -1 * prototypes.norm(p=2, dim=-1) ** 2
                reattach_names_values["head.bias"] = bias
                fast_names_values["head.bias"] = bias.detach().requires_grad_(True)
        return fast_names_values, reattach_names_values

    def _inner_forward(self, x_s: torch.Tensor, y_s: torch.Tensor, n_steps: int, x_q: torch.Tensor, aug_func: callable, source_label: bool, y_q: torch.Tensor = None):
        logs_dict = defaultdict(list)
        fast_phis = self.delta_hypernet.get_fast_params()
        fast_log_alphas_collections = self.mask_hypernet.get_fast_params(source_label=source_label)

        # sample a merged mask
        self.curr_temperature = max(0.5, 5 - self.num_metaupdates.item() / 4000)
        task_embed, merged_masks, merged_masks_sparsity, sep_mask_sparsities, \
            soft_selection_scores, selection_scores, dot_attn_scores, union_loss = self.mask_hypernet(self.net,
                                                                                                      x_s,
                                                                                                      y_s,
                                                                                                      None,
                                                                                                      fast_log_alphas_collections,
                                                                                                      temperature=self.curr_temperature)
        logs_dict["train/sparsity"].append(merged_masks_sparsity.detach())
        logs_dict["mask_selection"].append(selection_scores.detach())
        logs_dict["mask_attn"].append(dot_attn_scores.detach())
        logs_dict["mask_sparsities"].append(sep_mask_sparsities.detach())

        delta_params = self.delta_hypernet(phis=fast_phis)
        inteporated_outer_params = self._interpolate_params(masks=merged_masks, delta_params=delta_params)
        if self.nonparametric_head:
            fast_names_values, reattach_names_values = self._config_head(x_s=x_s,
                                                                         y_s=y_s,
                                                                         fast_weight_init=inteporated_outer_params)
            inteporated_outer_params.update(fast_names_values)
        # support classification loss
        y_s_pred = functional_call(self.net, inteporated_outer_params, (x_s,))
        support_loss = F.cross_entropy(input=y_s_pred, target=y_s)
        # take one inner loop during training only, or disable for faster training
        if self.training: 
            grads = torch.autograd.grad(support_loss, inteporated_outer_params.values(), create_graph=False, retain_graph=False)
            for np, g in zip(inteporated_outer_params.items(), grads):
                n, p = np
                inteporated_outer_params[n]  = p - self.args.meta_learner.inner_lr.lr * g
    
        if self.nonparametric_head:
            for n, clone_p in reattach_names_values.items():
                inteporated_outer_params[n] = inteporated_outer_params[n] + clone_p - clone_p.detach()

        # logs_dict["y_s_pred"].append(y_s_pred.detach()) #TODO: dont include this in meta-training; otherwise this will cause an issue in logging to weighst and bias
        logs_dict["train/support_ce"].append(support_loss.detach())
        logs_dict["train/support_accuracy"].append(accuracy(predictions=y_s_pred, targets=y_s))
        # logger.info(" ".join(f"{x:.3f}" for x in soft_selection_scores.detach().cpu().tolist()))
        return inteporated_outer_params, fast_log_alphas_collections, delta_params, soft_selection_scores, dot_attn_scores, task_embed, union_loss, logs_dict

    def _convert_id_to_mask_selection_scores(self, mask_id, device):
        b_int = [int(x) for x in f'{int(mask_id):08b}']
        scores = torch.tensor(b_int, device=device)
        return scores[:self.num_masks]

    def _convert_selection_scores_to_idx(self, scores):
        idx = 0
        for i, s in enumerate(scores):
            idx += s.detach() * 2 ** (self.num_masks - i - 1)
        return idx.long()

    def forward(self, x_s, y_s, x_q, y_q, **kwargs) -> dict:
        aug_func = kwargs.get("aug_func")
        self.mask_hypernet.clip_log_alpha()
        # inner 
        fast_net_params, fast_log_alphas_collections, delta_params, soft_selection_scores, \
            dot_attn_scores, task_embed, union_loss, inner_logs = self._inner_forward(x_s,
                                                                                      y_s,
                                                                                      self.args.meta_learner.num_inner_steps,
                                                                                      x_q=x_q,
                                                                                      aug_func=aug_func,
                                                                                      source_label=-1,
                                                                                      y_q=y_q)
        # outer 
        y_q_pred = functional_call(self.net, fast_net_params, (x_q,))

        finetune_mode = kwargs.get("finetune_mode", None)  
        if y_q is None or finetune_mode:
            inner_logs["y_q_pred"].append(y_q_pred.detach())
            if finetune_mode and self.args.meta_learner.inner_lr.lr:
                ####################################
                # if adaptation on the support:  ###
                ####################################
                with torch.no_grad():
                    detached_log_alphas_collections =  {n: l.detach() for n, l in fast_log_alphas_collections.items()}
                    mask_selection_scores = inner_logs["mask_selection"][-1].detach()
                    merged_masks, merged_mask_sparsity, _, _ = self.mask_hypernet.merge_masks_with_scores(detached_log_alphas_collections, mask_selection_scores)
                # get trainable parameters
                finetune_merged_masks = merged_masks
                finetune_delta_params = delta_params
                finetune_selection_scores = mask_selection_scores

                protomaml = False

                trainable_parameters = []
                if finetune_mode == "full":
                    # collapse into a single model
                    finetune_delta_params = self._interpolate_params(masks=finetune_merged_masks, delta_params=finetune_delta_params) 
                    finetune_delta_params = {n: p.detach().clone().requires_grad_(True) for n, p in finetune_delta_params.items()}
                    trainable_parameters.append({'params': finetune_delta_params.values(), 'lr': self.args.meta_learner.inner_lr.lr})
                    def _get_fast_params(m, p, s):
                        return p
                elif finetune_mode == "lora":
                    finetune_delta_params = self._interpolate_params(masks=finetune_merged_masks, delta_params=finetune_delta_params) 
                    self.net.load_state_dict({n:x.detach() for n,x in finetune_delta_params.items() if "head" not in n}, strict=False)
                    from peft import LoraConfig, get_peft_model
                    lora_config = LoraConfig(lora_alpha=32,
                                    lora_dropout=0.1,
                                    r=8,
                                    bias="none",
                                    target_modules=["query", "value"],
                                    inference_mode=False)
                    self.net = get_peft_model(self.net, lora_config)
                    print("lora has been attached")
                    finetune_delta_params = {n: p for n, p in self.net.named_parameters() if p.requires_grad}
                    trainable_parameters.append({'params': finetune_delta_params.values(), 'lr': self.args.meta_learner.inner_lr.lr * 10}) # use larger lrs for loRA
                    # self.net.update({"base_model.model."+ n:x.detach() for n,x in fast_net_params.items(()})
                    def _get_fast_params(m, p, s):
                        return p
                elif finetune_mode =='pretrained+full':  # these use pre-trained vit instead of smat
                    # collapse into a single model
                    finetune_delta_params = {n: p.detach().clone().requires_grad_(True) for n, p in self.net.named_parameters()}
                    trainable_parameters.append({'params': finetune_delta_params.values(), 'lr': self.args.meta_learner.inner_lr.lr})
                    def _get_fast_params(m, p, s):
                        return p
                elif finetune_mode =="pretrained+lora":
                    from peft import LoraConfig, get_peft_model
                    lora_config = LoraConfig(lora_alpha=32,
                                    lora_dropout=0.1,
                                    r=8,
                                    bias="none",
                                    target_modules=["query", "value"],
                                    inference_mode=False)
                    self.net = get_peft_model(self.net, lora_config, "default")
                    print("lora has been attached")
                    finetune_delta_params = {n: p for n, p in self.net.named_parameters() if p.requires_grad}
                    trainable_parameters.append({'params': finetune_delta_params.values(), 'lr': self.args.meta_learner.inner_lr.lr * 10})
                    def _get_fast_params(m, p, s):
                        return p
                else:
                    raise KeyError(finetune_mode)
                
                criterion = torch.nn.CrossEntropyLoss()
                _opt = torch.optim.AdamW(trainable_parameters, lr=self.args.meta_learner.inner_lr.lr)
            
                #####################
                ## optimization steps
                #####################
                for finetune_step in range(self.args.meta_learner.num_inner_steps + 1):
                    teach_params = _get_fast_params(m=finetune_merged_masks, p=finetune_delta_params, s=finetune_selection_scores)
                    if self.nonparametric_head and not protomaml:
                        _, reattach_names_values = self._config_head(x_s=x_s,
                                                                        y_s=y_s,
                                                                        fast_weight_init=teach_params)
                    else:
                        raise NotImplementedError
                    # record query prediction:
                    with torch.no_grad():
                        y_q_f =  functional_call(self.net, teach_params, (x_q,True))
                        target_y_q_pred = F.cosine_similarity(y_q_f[:,None,:],
                                                              reattach_names_values["head.weight"][None,:,:],
                                                              dim=-1,eps=1e-12) * 10
                        inner_logs["y_q_pred"].append(target_y_q_pred.detach())
                    if finetune_step == self.args.meta_learner.num_inner_steps:
                        break
                    # optimize on aug support set
                    y_s_f = functional_call(self.net, teach_params, (aug_func(x_s),True))
                    target_y_s_pred = F.cosine_similarity(y_s_f[:,None,:],
                                                          reattach_names_values["head.weight"][None,:,:],
                                                        dim=-1,eps=1e-12) * 10
                    finetune_loss = criterion(input=target_y_s_pred, target=y_s)
                    _opt.zero_grad()
                    finetune_loss.backward()
                    _opt.step()
                if "lora" in finetune_mode:
                    self.net =deepcopy(self.pretrained_state_dict)
                    print("lora has been deleted")           
            return inner_logs

        ############################
        ###  if meta-training    ###
        ############################
        teach_T = self.args.meta_learner.dense_teacher.T
        teach_lr = self.args.meta_learner.dense_teacher.lr
        teach_lam = self.args.meta_learner.dense_teacher.lam
        if not self.training:
            logger.warning(f"entered optimization loop with model.eval()")
        else:
            self.num_metaupdates += 1

        #####################
        ### teacher model ###
        #####################
        teach_params = {n: p.detach().clone().requires_grad_(True) for n, p in fast_net_params.items()}
        max_teach_steps = 1
        for teach_step in range(max_teach_steps + 1):
            target_y_q_pred = functional_call(self.net, teach_params, (x_q,), strict=True)
            if teach_step == max_teach_steps:
                teach_query_acc = accuracy(target_y_q_pred.detach(), y_q) # - accuracy(y_q_pred.detach(), y_q)
                break
            _loss = F.cross_entropy(input=target_y_q_pred, target=y_q)
            grads = torch.autograd.grad(_loss, list(teach_params.values()))
            for n, g in zip(teach_params.keys(), grads):
                teach_params[n] = teach_params[n] - teach_lr * g
        target_selection_scores = self._convert_id_to_mask_selection_scores(0, device=x_s.device)
        with torch.no_grad():
            pre_parmas = {n: p.detach() for n, p in self.net.named_parameters()}
            pre_head, _  = self._config_head(x_s, y_s, pre_parmas)
            pre_parmas.update(pre_head)
            pre_y_q_pred = functional_call(self.net, pre_parmas, (x_q,), strict=True)
      
        #########################
        #### collect losses #####
        #########################
        query_ce = F.cross_entropy(y_q_pred, y_q)
        query_kd = F.kl_div(F.log_softmax(target_y_q_pred.detach() / teach_T, dim=-1),
                            F.log_softmax(y_q_pred / teach_T, dim=-1),
                            reduction="batchmean",
                            log_target=True) * teach_T ** 2 * 0.5
        query_kd += F.kl_div(F.log_softmax(pre_y_q_pred.detach() / teach_T, dim=-1),
                            F.log_softmax(y_q_pred / teach_T, dim=-1),
                            reduction="batchmean", 
                            log_target=True) * teach_T ** 2 * 0.5
        prior_selection_loss = torch.pow(dot_attn_scores, 2).mean() * 1e-4 #penalize excessively large activation logits for stablized training
        query_acc = accuracy(y_q_pred, y_q)

        ####################################
        ### mask sparse and lp penalties ###
        ####################################
        mask_selection = inner_logs.pop("mask_selection")[-1]
        mask_sparsities = inner_logs.pop("mask_sparsities")[-1]
        mask_attn = inner_logs.pop("mask_attn")[-1]
        lagrangian = torch.zeros_like(query_ce)
        sparstiy_loss = torch.zeros_like(query_ce)
        expected_l0s = self.mask_hypernet.sparse_penalty(fast_log_alphas_collections)
        delta_sparsities = mask_sparsities - self.mask_sparsity_targets
        fill_idx = torch.greater(delta_sparsities, 0).float()
        self.lagrangian_multiplier.data = self.lagrangian_multiplier.data * (1 - fill_idx) + torch.zeros_like(self.lagrangian_multiplier) * fill_idx
        lagrangian = torch.sum(delta_sparsities * self.lagrangian_multiplier * mask_selection)
        sparstiy_loss = torch.sum((self.mask_sparsity_targets - 1 + expected_l0s) * self.lagrangian_multiplier.detach() * mask_selection)
        # expected_l2 = torch.zeros_like(query_ce)
        ####################################
        ###    return loss and logging ###
        ####################################
        res_dict = {
            "outer_loss": {"query_ce": query_ce * (1-teach_lam),
                           "query_kd": query_kd * teach_lam,
                           "sparsity_loss": sparstiy_loss,
                           "lagrangian": lagrangian,
                           "l2": prior_selection_loss},
            "train/query_accuracy/learner": query_acc,
            "train/query_accuracy/target": teach_query_acc
        }
        # clean-up and update the list attributes in inner_logs
        for k, v in inner_logs.items():
            if isinstance(v, list):
                last_v = v[-1].detach()
            res_dict.update({k: last_v})
        res_dict.update({f"train/lagrangian_multiplier_{idx}": lag.detach() for idx, lag in enumerate(self.lagrangian_multiplier)})
        res_dict.update({f"train/mask_selection_{idx}/learner": lag.detach() for idx, lag in enumerate(mask_selection)})
        res_dict.update({f"train/mask_selection_{idx}/target": lag.detach() for idx, lag in enumerate(target_selection_scores)})
        res_dict.update({f"train/mask_selection_idx/learner": self._convert_selection_scores_to_idx(mask_selection)})
        res_dict.update({f"train/mask_selection_idx/target": self._convert_selection_scores_to_idx(target_selection_scores)})
        res_dict.update({f"train/mask_sparsity_{idx}": lag.detach() for idx, lag in enumerate(mask_sparsities)})
        res_dict.update({f"train/mask_attn_{idx}": lag.detach() for idx, lag in enumerate(mask_attn)})
        # detach everything else except the loss terms
        for k, v in res_dict.items():
            if "outer_loss" not in k:
                res_dict[k] = v.item()
        # format display for pbar:
        log_msg = f'Q Acc: {res_dict["train/query_accuracy/learner"]:.3f} |Sparsity: {res_dict["train/sparsity"]:.3f}'
        res_dict["log_msg"] = log_msg
        return res_dict

    def metatrain_parameters(self, as_list=False):
        param_dict = [
            {'params': self.delta_hypernet.parameters(),
             'name': "delta_params",
             "weight_decay": 1e-4},
            {'params': self.mask_hypernet.get_trainable_mask_parameters(),
             "lr": self.args.meta_learner.outer_lr.mask_lr,
             "momentum": 0,
             "name": 'masks'},
            {'params': self.lagrangian_multiplier,
             "lr": self.args.meta_learner.outer_lr.lagrangian_lr,
             "betas": (0, 0),
             "momentum": 0,
             'name': 'lag_mul'}
        ]
        if self.args.meta_learner.inner_lr.learnable:
            param_dict.append({'params': self._lrs, "lr": self.args.meta_learner.outer_lr.inner_lr_lr, 'name': 'inner_lrs'})
        if as_list:
            param_list = []
            for param_group in param_dict:
                param_list += list(param_group["params"])
            return param_list
        else:
            return param_dict

    def clip_grad_parameters(self):
        yield from self.mask_hypernet.get_trainable_mask_parameters()

    def get_masked_basenet(self, mask_selection_scores):
        with torch.no_grad():
            log_alphas_collections = self.mask_hypernet.get_fast_params()
            merged_masks, merged_sparsity, mask_sparsities, union_loss = self.mask_hypernet.merge_masks_with_scores(log_alphas_collections=log_alphas_collections,
                                                       scores=mask_selection_scores)
            
            fast_phis = self.delta_hypernet.get_fast_params()
            delta_params = self.delta_hypernet(phis=fast_phis)
            net_params = self._interpolate_params(masks=merged_masks, delta_params=delta_params)
        return net_params
