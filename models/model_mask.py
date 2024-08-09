import functools
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from transformers import AutoModel
from timm.models import vision_transformer
from transformers.models.bert import modeling_bert

from utils import misc , pos_embed


class MRM_MASK(nn.Module):
        
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                 norm_layer=functools.partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False, 
                 decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=8,
                 global_project_dim=512, **kwargs):
        

        super().__init__()

        self.patch_embed = vision_transformer.PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            vision_transformer.Block(embed_dim, num_heads, mlp_ratio, 
                                     qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        self.scan_mlp = nn.Sequential(
                            nn.Linear(embed_dim, decoder_embed_dim, bias=True), 
                            norm_layer(decoder_embed_dim), nn.GELU(), 
                            nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
                                    torch.zeros(1, num_patches + 1, decoder_embed_dim), 
                                    requires_grad=True)

        self.decoder_blocks = nn.ModuleList([
            vision_transformer.Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, 
                                     qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size ** 2) * in_channels, bias=True)

        self.global_scan_mlp = nn.Linear(embed_dim, global_project_dim, bias=True)

        self.norm_pix_loss = norm_pix_loss
    
        self.initialize_weights()
    
    def initialize_weights(self):

        pos_embedding = pos_embed.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                                          int(self.patch_embed.num_patches**.5), 
                                                          cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embedding).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def gen_prior_masking(self, local_features, global_feature, prior_mask_ratio):

        N, L, D = local_features.shape  
        global_feature = global_feature.unsqueeze(1)  
        cos_sim = F.cosine_similarity(local_features, global_feature, dim=-1)  
        topk_val, topk_idx = torch.topk(cos_sim, k=int(L*(1-prior_mask_ratio)), dim=1, largest=True, sorted=True)

        mask = torch.ones_like(cos_sim)
        mask.scatter_(dim=1, index=topk_idx, value=0)

        return mask
        
    def forward_scan_encoder(self, x, is_training=False):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            if is_training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        
        return x

    def forward(self, batch, prior_mask_ratio=0.5, is_training=False, loss_keys=[]):
        
        device = next(self.parameters()).device

        imgs= batch["image"].to(device)
        scan_latents = self.forward_scan_encoder(imgs, is_training=is_training) 

        local_features = self.global_scan_mlp(scan_latents[:, 1:, :]) 
        global_scan_features = local_features.max(dim=1)[0]   

        mask = self.gen_prior_masking(local_features, global_scan_features, prior_mask_ratio=prior_mask_ratio)

        if is_training:
            raise ValueError("MASK 模型仅用于推理！")
        else:
            return {  
                    'MASK':{'scan_latents':scan_latents,
                            'mask':mask
                    }
                }
