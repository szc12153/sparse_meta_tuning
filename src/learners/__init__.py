import inspect
import torch.optim as optim
from .base_learners import get_base_learner
from .meta_learners import get_meta_learner
from dataclasses import asdict
import torch.optim.lr_scheduler as lr_scheduler
from timm.scheduler import create_scheduler

def get_optimizer(parameters, args):
    kwargs = asdict(args)
    optimizer = optim.__dict__.get(kwargs["name"],None)
    assert optimizer, f'{kwargs["name"]} optimizer undefined'
    valid_keys = inspect.getfullargspec(optimizer.__init__).args
    for key in list(kwargs.keys()):
        if key not in valid_keys:
            kwargs.pop(key)
    return optimizer(parameters, **kwargs)


def get_scheduler(optimizer, args):
    if not args.sched:
        return None
    from transformers.optimization import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=5,
                                                num_training_steps=50)
    return scheduler
    # return create_scheduler(args,optimizer)[0]

def _get_scheduler(optimzier, args):
    kwargs = asdict(args)
    scheduler = lr_scheduler.__dict__.get(kwargs["name"], None)
    if not scheduler:
        return None
    valid_keys = inspect.getfullargspec(scheduler.__init__).args
    for key in list(kwargs.keys()):
        if key not in valid_keys:
            kwargs.pop(key)
    return scheduler(optimzier, **kwargs)