import torch
import torch.nn as nn
from .common import BaseLearner
from typing import Union


class VisionTransformer(BaseLearner):
    def _get_backbone(self,args)->nn.Module:
        from transformers import CLIPVisionModel, ViTModel
        if args.base_learner.backbone=="ViT":
            encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            encoder.pooler = nn.Identity()
        elif args.base_learner.backbone=="CLIPViT":
            encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            encoder.vision_model.post_layernorm = nn.Identity()
        elif args.base_learner.backbone =="ViT_Small_1k":
            encoder = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
            encoder.pooler = nn.Identity()
        elif args.base_learner.backbone =="ViT_Base_21k":
            encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            encoder.pooler = None
        elif args.base_learner.backbone =="ViT_Small_DINO":
            encoder = ViTModel.from_pretrained('facebook/dino-vits16')
            encoder.pooler = None
        else:
            raise NotImplementedError(f"backbone : {args.base_learner.backbone} is not implemented")
        return encoder

    def _forward_encoder(self, images)->torch.Tensor:
        cls_token = self.encoder(images, interpolate_pos_encoding=True).last_hidden_state[:, 0, :]
        return cls_token


class TorchHubTransformer(BaseLearner):
    def _get_backbone(self, args) ->nn.Module:
        if args.base_learner.backbone== "ViT_Small_DINO_torchhub":
            import src.learners.base_learners.torchhub_vit as TorchHubViTModels
            encoder = TorchHubViTModels.__dict__['vit_small'](patch_size=16, num_classes=0, devices=args.devices)
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            encoder.load_state_dict(state_dict, strict=True)
        elif args.base_learner.backbone== "ViT_Base_DINO_torchhub":
            import src.learners.base_learners.torchhub_vit as TorchHubViTModels
            encoder = TorchHubViTModels.__dict__['vit_base'](patch_size=16, num_classes=0, devices=args.devices)
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            encoder.load_state_dict(state_dict, strict=True)
            print('Pretrained weights found at {}'.format(url))
        elif args.base_learner.backbone=="ViT_Base_21k_torchhub":
            import src.learners.base_learners.google_vit as GoogleViTModels
            import os
            import numpy as np
            encoder = GoogleViTModels.__dict__['vit_base'](devices=args.devices)
            url = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
            pretrained_weights = 'pretrained_ckpts/vit_base_patch16_224_in21k.npz'
            if not os.path.exists(pretrained_weights):
                try:
                    import wget
                    os.makedirs('pretrained_ckpts', exist_ok=True)
                    wget.download(url, pretrained_weights)
                except:
                    print(f'Cannot download pretrained weights from {url}. Check if `pip install wget` works.')
            encoder.load_from(np.load(pretrained_weights))
            print('Pretrained weights found at {}'.format(url))
        elif args.base_learner.backbone=="ViT_Large_21k_torchhub":
            from src.learners.base_learners.google_vit import GoogleViTModels, CONFIGS
            import os
            import numpy as np

            config = CONFIGS['ViT-L_16']
            encoder = GoogleViTModels(config, 224)

            url = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz'
            pretrained_weights = 'pretrained_ckpts/vit_large_patch16_224_in21k.npz'

            if not os.path.exists(pretrained_weights):
                try:
                    import wget
                    os.makedirs('pretrained_ckpts', exist_ok=True)
                    wget.download(url, pretrained_weights)
                except:
                    print(f'Cannot download pretrained weights from {url}. Check if `pip install wget` works.')

            encoder.load_from(np.load(pretrained_weights))
            print('Pretrained weights found at {}'.format(pretrained_weights))

        else:
            raise NotImplementedError(f"backbone : {args.base_learner.backbone} is not implemented")
        return encoder

    def _forward_encoder(self,images) -> torch.Tensor:
        return self.encoder(images)


def get_vit_model(args):
    if 'torchhub' in args.base_learner.backbone:
        return TorchHubTransformer(args)
    else:
        return VisionTransformer(args)