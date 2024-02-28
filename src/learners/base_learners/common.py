import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from src.utils import Configuration, get_logger

logger = get_logger(__name__)


def sim_func(dist:str):
    """x : batch, feat_dim"""
    """y: n_classes, feat_dim"""
    if dist == "cosine":
        def _cosine_sim(x,y,z=None):
            d = nn.functional.cosine_similarity(x[:,None,:],y[None,:,:],dim=-1,eps=1e-12) * 10
            if z is not None:
                d += z * 0
            return  d
        return _cosine_sim
    if dist == "euclidean":
        def _euclidean_sim(x,y,z):
            return -torch.norm(x[:,None,:]-y[None,:,:],p="fro",dim=-1)+ z * 0
        return _euclidean_sim
    if dist == "dot":
        def _dot_sim(x, y,z):
            return torch.mm(x,y.T) + z * 0
        return _dot_sim
    if dist == "affine":
        def _affine_sim(x,y,z):
            return F.linear(x,y,z)
        return _affine_sim
    raise NotImplementedError(f"non-parametric head : {dist} is not implemented")


class BaseLearner(nn.Module):
    def __init__(self,args:Configuration):
        nn.Module.__init__(self)
        self.encoder = self._get_backbone(args)
        logger.info(f"ViT backbone : {args.base_learner.backbone}")
        # get feat_dim
        with torch.no_grad():
            cls_token = self._forward_encoder(torch.randn(1, 3,224,224))
            self._feat_dim = cls_token.size(-1)
        # get head
        if args.base_learner.head == "linear":
            self.head = nn.Linear( self._feat_dim, args.base_learner.output_dim, bias = args.base_learner.bias)
            self.nonparametric_head = False
        else:
            logger.info(f"using NON-PARAMETRIC head with {args.base_learner.head} distance")
            self.nonparametric_head = True
            self.head_dist_func = sim_func(args.base_learner.head)
            # place holder, so masks and lr initialize properly, however this neglect dim-scale in the sparse penalty
            self.head = nn.Linear(self._feat_dim, 1, bias = args.base_learner.bias)
            # self.logits_bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
            # self.logits_scale = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        # for meta-training only
        if args.meta_learner is not None:
            logger.info("init head in base model with zeros")
            for p in self.head.parameters():
                nn.init.zeros_(p)
            # freeze embeddings layers for finetuning
            if args.meta_learner.freeze_inner:
                self.freeze_inner = set(args.meta_learner.freeze_inner)
            else:
                self.freeze_inner = None
            if args.meta_learner.freeze_outer:
                self.freeze_inner.union(set(args.meta_learner.freeze_inner))
                self.freeze_outer = set(args.meta_learner.freeze_outer)
            else:
                self.freeze_outer = None
            # debug logging
            inner_params = [_[0] for _ in self.inner_meta_params()]
            outer_params = [_[0] for _ in self.outer_meta_params()]
            logger.debug(f"{'Parameter':<60}: {'F_I':<5} {'F_O':<5}")
            for n, p in self.named_parameters():
                freeze_inner, freeze_outer = "Y","Y"
                if n in inner_params:
                    freeze_inner = ""
                if n in outer_params:
                    freeze_outer = ""
                logger.debug(f"{n:<60}: {freeze_inner:<5} {freeze_outer:<5}")
        p_count = 0
        p_count_on_aprior= 0
        for n, p in self.named_parameters():
            p_count += p.numel()
            if any(k in n for k in ["head","bias","layernorm"]) and "embeddings" not in n:
                p_count_on_aprior+=p.numel()
        logger.info(f"Total Number of Parameters in the Backbone :{p_count}")
        logger.info(f"Percentage of on-a-prior Parameters in the Backbone :{p_count_on_aprior/p_count*100:.3f}")


    @property
    def input_size(self):
        return 224

    @property
    def feat_dim(self):
        return self._feat_dim

    @abstractmethod
    def _get_backbone(self, args)->nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _forward_encoder(self,images)->torch.Tensor:
        raise NotImplementedError

    def forward(self, images, return_features_only=False):
        features = self._forward_encoder(images) # batch_size, feat_dim
        if return_features_only:
            return features
        else:
            if self.nonparametric_head:
                logits = self.head_dist_func(x=features,y=self.head.weight,z=self.head.bias)
            else:
                logits = self.head(features)
            return logits

    def outer_meta_params(self):
        for n,p in self.named_parameters():
            if not self.freeze_outer or not any([k in n for k in self.freeze_outer]):
              yield (n, p)

    def inner_meta_params(self):
        for n, p in self.named_parameters():
            if not self.freeze_inner or not any([k in n for k in self.freeze_inner]):
                yield (n, p)
