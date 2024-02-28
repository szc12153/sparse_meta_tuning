from dataclasses import dataclass, asdict
from typing import List,Optional,Tuple, Union

@dataclass
class Optimizer:
    name: str
    lr: float
    weight_deacy: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9
    nesterov: bool = False
    max_grad_norm: float = -1.

@dataclass
class Scheduler:
    sched: Optional[str]
    warmup_lr: Optional[float]
    min_lr: float
    decay_epochs: int
    warmup_epochs: int
    cooldown_epochs: int
    patience_epochs: int
    decay_rate: float
    lr_noise: Union[List[float],None] = None
    lr_noise_pct: float=0.67
    lr_noise_std: float=1.0


@dataclass
class Configuration:

    @dataclass
    class MetaLearner:
        @dataclass
        class OuterLR:
            lr: float # default outer-loop lr
            inner_lr_lr: Optional[float]
            mask_lr: Optional[float]
            lagrangian_lr: Optional[float]
        @dataclass
        class InnerLR:
            lr: float
            structured: bool
            learnable: bool
            mask_inner_lr_multiplier: Optional[int] # a scalar multiplier for the mask inner-loop gradients
            per_param_group: bool # one lr per param_group in model.parameters()
        @dataclass
        class Sparsity:
            structured: bool # input sparsity for linear and output sparsity for conv, no sparsity for the bias
            on_bias: bool
            target: float # targted model sparsity for training
            sparsify_where: List[str] # sparsify the params or grads, only sp_param is used for now
            penalty_strength: Optional[float] # only used for mscn-param with penalty, not used in mscn-param lagragian
            update_mask: Optional[bool] # whether to update the masks during the inner-loop
            straight_through_relu: Optional[bool] # ste for sparsemaml inner leraning rates
            sample_mask_per_step: bool # whether to sample a mask in each inner loop step for mscn
            stochastic_mask: bool
            fix_embeddings: bool
            num_experts: int
        @dataclass
        class DenseTeacher:
            num_steps: int
            T: float
            lam: float
            lr: float

        model: str
        first_order: bool
        num_inner_steps: int
        zero_trick: bool # whether to initialize the final non-paramteric classifier with zero, if false then use initialize using class prototypes
        freeze_inner: Union[List[str],None]  # parameters to freeze during the inner-loop adaptation
        freeze_outer: Union[List[str],None]  # parameters to freeze during meta-training
        inner_lr: InnerLR
        outer_lr: OuterLR
        sparsity: Optional[Sparsity]
        dense_teacher:  Optional[DenseTeacher]

    @dataclass
    class BaseLearner:
        backbone: str
        head: str  # e.g. cosine, euclidean, dot etc..
        feature_dim: int # not used for VIT backbone
        bias: bool # whether to include bias in the final classifier
        output_dim: int # the output dim of the final classifier, not used if using non-parameteric classifer
        keep_bn_stats: bool # whether to keep or discard the pre-trained BN statistics in the backbone; does not affect ViT

    @dataclass
    class Training:
        datasets: List[str]
        finetune: str
        max_patience: int
        max_epoch: int
        num_ways: int
        num_shots: int
        num_queries: int
        batchsize: int
        fp16: bool  # whether to use mixed precision
        gradient_surgery: bool # whether to apply gradient_surgery when back-propagating the losses
        image_size: int # 128 or 224
        source_info: bool

    @dataclass
    class Testing:
        datasets: List[str]
        max_tasks: int
        num_ways: int
        num_shots: int
        num_queries: int

    @dataclass
    class Logging:
        print_freq:int
        val_freq: int
        save_freq: int
        exp_dir: str

    @dataclass
    class Graft:
        datasets: List[str]
        batchsize: int
        max_iter: int
        num_shots: int
        init_sparsity: float
        sigmoid_bias: float
        optimizer: Optimizer

    @dataclass
    class Finetune:
        datasets: List[str]
        batchsize: int
        max_epoch: int
        max_iter: int
        num_shots: int
        max_patience: int
        optimizer: Optimizer
        scheduler: Optional[Scheduler]

    base_learner: BaseLearner
    meta_learner: Optional[MetaLearner]
    optimizer: Optional[Optimizer]
    scheduler: Optional[Scheduler]
    train: Optional[Training]
    test: Optional[Testing]
    logging: Logging
    graft: Optional[Graft]
    finetune: Optional[Finetune]
