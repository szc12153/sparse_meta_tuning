from typing import Mapping, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import  get_logger, Configuration
from copy import deepcopy
from tqdm import tqdm, trange
from itertools import chain
from torch.func import functional_call, vmap, grad


logger = get_logger(__name__)

class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=False)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=False)

        # backbone
        self.backbone = backbone
        for p in self.finetune_parameters():
            p.requires_grad_(True)

    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim()-1, eps=1e-12)

        cls_scores = f @ w.T # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    def finetune_parameters(self):
        yield from self.backbone.parameters()

    def create_prototypes(self,x_s,y_s,num_classes):
        with torch.no_grad():
            nSupp, C, H, W = x_s.shape
            supp_f = self.backbone.forward(x_s)
            supp_f = supp_f.view(nSupp, -1)

            supp_y_1hot = F.one_hot(y_s, num_classes).T  # nC, nSupp

            # B, nC, nSupp x B, nSupp, d = B, nC, d
            prototypes = torch.mm(supp_y_1hot.float(), supp_f)
            prototypes = prototypes / supp_y_1hot.sum(dim=1, keepdim=True)  # NOTE: may div 0 if some classes got 0 images
        self.prototypes = torch.nn.Parameter(prototypes, requires_grad=True)

    def forward(self, x_s, y_s, x_q, return_q_feat_only = False):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        if return_q_feat_only:
            return self.backbone.forward(x_q)
        num_classes = y_s.max() + 1 # NOTE: assume B==1

        nSupp, C, H, W = x_s.shape
        supp_f = self.backbone.forward(x_s)
        supp_f = supp_f.view(nSupp, -1)

        supp_y_1hot = F.one_hot(y_s, num_classes).T #nC, nSupp

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        prototypes = torch.mm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=1, keepdim=True) # NOTE: may div 0 if some classes got 0 images

        feat = self.backbone.forward(x_q)
        # feat = feat.view(B, x_q.shape[1], -1) # B, nQry, d

        logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
        return logits


class LoRAPMF(ProtoNet):
    def __init__(self, backbone):
        super().__init__(backbone)
        from peft import LoraConfig, get_peft_model
        self.lora_config = LoraConfig(lora_alpha=32,
                                      lora_dropout=0.1,
                                      r=8,
                                      bias="none",
                                      target_modules=["query", "value"],
                                      inference_mode=False)

        #  backbone + lora
        self.backbone = get_peft_model(backbone,self.lora_config)
        # map torchhub pre-trained weights to huggingface weights, different names
        self.torchhub_to_huggingface_keywords_mapping={
            "backbone." : "backbone.base_model.model.",
            "blocks.": "encoder.layer.",
            "mlp.fc1.": "intermediate.dense.",
            "mlp.fc2.": "output.dense.",
            "norm1.": "layernorm_before.",
            "norm2.": "layernorm_after.",
            ".norm.": ".layernorm.",
            "attn.proj.": "attention.output.dense.",
            ".cls_token": ".embeddings.cls_token",
            ".pos_embed":".embeddings.position_embeddings",
            "patch_embed.proj.":"embeddings.patch_embeddings.projection."
        }
        self.backbone.float()

    def finetune_parameters(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                yield p

    def load_state_dict(self, state_dict, strict=False):
        import sys
        try:
            super().load_state_dict(state_dict=state_dict, strict=True)
        except RuntimeError:
            converted_state_dict={}
            for n,p in state_dict.items():
                for hub_name, hug_name in self.torchhub_to_huggingface_keywords_mapping.items():
                    if hub_name in n:
                        n = n.replace(hub_name,hug_name)
                if "attn.qkv." in n:
                    if p.dim()==2:
                        q,k,v = p.reshape(3, p.size(-1), p.size(-1)).unbind(0)
                    else:
                        q,k,v = p.reshape(3,-1).unbind(0)
                    converted_state_dict[n.replace("attn.qkv.","attention.attention.query.base_layer.")] = q
                    converted_state_dict[n.replace("attn.qkv.","attention.attention.key.")] = k
                    converted_state_dict[n.replace("attn.qkv.","attention.attention.value.base_layer.")] = v
                else:
                    converted_state_dict[n] = p
        
            missing, unexpetced = super().load_state_dict(converted_state_dict, False)
            assert not unexpetced, unexpetced
            assert all("lora" in k for k in missing), missing
            for n, p in self.named_parameters():
                if "lora_" in n:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        # sys.exit("okay")
        
    def create_prototypes(self, x_s, y_s, num_classes):
        with torch.no_grad():
            num_classes = y_s.max() + 1  # NOTE: assume B==1

            nSupp, C, H, W = x_s.shape
            supp_f = self.backbone.forward(x_s, interpolate_pos_encoding=True).last_hidden_state[:, 0, :]
            supp_f = supp_f.view(nSupp, -1)

            supp_y_1hot = F.one_hot(y_s, num_classes).T  # nC, nSupp
            # B, nC, nSupp x B, nSupp, d = B, nC, d
            prototypes = torch.mm(supp_y_1hot.float(), supp_f)
            prototypes = prototypes / supp_y_1hot.sum(dim=1,
                                                      keepdim=True)  # NOTE: may div 0 if some classes got 0 images
        self.prototypes = torch.nn.Parameter(prototypes, requires_grad=True)

    def forward(self, x_s, y_s, x_q, return_q_feat_only = False):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        if return_q_feat_only:
            return self.backbone(x_q,interpolate_pos_encoding=True).last_hidden_state[:, 0, :]
        num_classes = y_s.max() + 1  # NOTE: assume B==1

        nSupp, C, H, W = x_s.shape
        supp_f = self.backbone.forward(x_s,interpolate_pos_encoding=True).last_hidden_state[:, 0, :]
        supp_f = supp_f.view(nSupp, -1)

        supp_y_1hot = F.one_hot(y_s, num_classes).T  # nC, nSupp

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        prototypes = torch.mm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=1, keepdim=True)  # NOTE: may div 0 if some classes got 0 images

        feat = self.backbone.forward(x_q,interpolate_pos_encoding=True).last_hidden_state[:, 0, :]
        # feat = feat.view(B, x_q.shape[1], -1) # B, nQry, d

        logits = self.cos_classifier(prototypes, feat)  # B, nQry, nC
        return logits
    
    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim() - 1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim() - 1, eps=1e-12)

        cls_scores = f @ w.T  # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores
        

def get_finetune_model(finetune):
    if finetune == "full":
        BaseClass = ProtoNet
    elif finetune == "lora":
        BaseClass = LoRAPMF
    else:
        raise NotImplementedError, finetune

    class Finetune(BaseClass):
        def __init__(self, net: nn.Module, args: Configuration):
            super().__init__(net)
            self.num_iters = args.meta_learner.num_inner_steps
            self.lr = args.meta_learner.inner_lr.lr
            self.prototypes = None

        def load_state_dict(self, state_dict, strict=True, echo_name=None):
            super().load_state_dict(state_dict, strict)
            state_dict = self.backbone.state_dict()
            self.backbone_state = deepcopy(state_dict)
            logger.warning(f"loaded full model state_dict and saved pre-trained backbone.state_dict, echo {echo_name}")

        def predict(self,x_s,y_s,x_q, protomaml):
            if not protomaml:
                return super().forward(x_s, y_s, x_q)
            else:
                raise NotImplementedError
                q_feat = super().forward(x_s,y_s,x_q,return_q_feat_only=True)
                return self.cos_classifier(self.prototypes, q_feat)

        def forward(self, x_s, y_s, x_q, **kwargs):
            """ optimzie on support, then make prediction on the query set"""
            aug_func = kwargs.get("aug_func")
            if not aug_func:
                aug_func = lambda x: x
            criterion = nn.CrossEntropyLoss()
            # reset backbone to pre-trained state-dict
            self.backbone.load_state_dict(self.backbone_state, strict=True)
            use_protomaml = False
            if use_protomaml:
                num_classes = y_s.max() + 1
                self.create_prototypes(x_s, y_s, num_classes)
                opt = torch.optim.Adam(list(self.finetune_parameters()) + [self.prototypes],
                                       lr=self.lr,
                                       betas=(0.9, 0.999),
                                       weight_decay=0.)
            else:
                # try:
                #     opt = super().get_optimizer(self.lr)
                # except AttributeError:
                opt = torch.optim.Adam(list(self.finetune_parameters()), lr=self.lr)
            _loss_scaler = torch.cuda.amp.GradScaler(init_scale=2**16) 
            rest_dict = {"y_q_pred": []}
            # finetuning on the support
            pbar = trange(self.num_iters, leave=False)
            if self.lr != 0:
                for i in pbar:
                    with torch.cuda.amp.autocast(enabled=True):
                        # query prediction
                        with torch.no_grad():
                            y_q_pred = self.predict(x_s, y_s, x_q, protomaml=use_protomaml)
                            rest_dict["y_q_pred"].append(y_q_pred.detach())
                        logits = self.predict(x_s, y_s, aug_func(x_s), protomaml=use_protomaml)
                        loss = criterion(logits, y_s)
                    opt.zero_grad()
                    _loss_scaler.scale(loss).backward()
                    _loss_scaler.step(opt)
                    _loss_scaler.update()
                    s_acc = (logits.argmax(dim=-1) == y_s).float().mean().item() * 100
                    pbar.set_description(f"Episode {i}, LR {self.lr:.6f} Acc {s_acc:.2f} S {len(x_s)} Q {len(x_s)}")
            # final prediction
            with torch.no_grad():
                y_q_pred = self.predict(x_s, y_s, x_q, protomaml=use_protomaml)
                rest_dict["y_q_pred"].append(y_q_pred.detach())
            return rest_dict

    return Finetune


