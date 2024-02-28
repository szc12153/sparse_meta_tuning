from .conv4 import get_cov4_model
from .vit import get_vit_model
from .resnet import get_resnet_model


def get_base_learner(args):
    if "ViT" in args.base_learner.backbone:
        return get_vit_model(args)
    if args.base_learner.backbone in ["CONV4"]:
        return get_cov4_model(args)
    if args.base_learner.backbone in ["RESNET18"]:
        return get_resnet_model(args)
    else:
        raise Exception
