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

from .language_model import * 
from .Trans_language_model import EncoderDecoder

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


class MRM_GEN(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                 norm_layer=functools.partial(nn.LayerNorm, eps=1e-6), tokenizer=None, max_caption_length=100, norm_pix_loss=False, 
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

        self.language_model = EncoderDecoder(tokenizer, max_caption_length)

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

    def patchify(self, imgs):


        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def random_masking(self, x, guide_mask, mask_ratio):

        N, L, D = x.shape  
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  

        noise[guide_mask == 0] += 0.8   
        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
 
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_scan_encoder(self, x, guide_mask, mask_ratio=0.5, is_training=True):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, guide_mask, mask_ratio)
        else:
            mask, ids_restore = None, None

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            if is_training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore

    def forward_scan_decoder(self, x, ids_restore, is_training=True):

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) 
        x = torch.cat([x[:, :1, :], x_], dim=1) 
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            if is_training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        x = x[:, 1:, :]

        return x

    def reconstruction_loss(self, imgs, pred, mask):

        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  

        loss = (loss * mask).sum() / mask.sum()  
        return loss

    def forward(self, batch, guide_mask, mask_ratio=0.5, is_training=True, loss_keys=[], max_caption_length=100):
        
        device = next(self.parameters()).device

        imgs= batch["image"].to(device)

        ids, attention_mask = batch["ids"].to(device), batch["attention_mask"].to(device)

        if is_training:

            output_loss = {}

            if 'msm' in loss_keys:
                scan_latents, mask, ids_restore = self.forward_scan_encoder(imgs, guide_mask=guide_mask, mask_ratio=mask_ratio, is_training=is_training)

                scan_features = self.scan_mlp(scan_latents)
                scan_pred = self.forward_scan_decoder(scan_features, ids_restore, is_training=is_training) # torch.Size([32, 196, 768])
                masked_im_loss = self.reconstruction_loss(imgs, scan_pred, mask)
                output_loss['msm'] = masked_im_loss

            if 'gen' in loss_keys:

                gen_img_latents, _, _ = self.forward_scan_encoder(imgs, None, mask_ratio=0.0, is_training=is_training)

                gen_img_latents = gen_img_latents[:, 1:, :]   

                outputs, _, _, _ = self.language_model(fc_feats=None, att_feats=gen_img_latents, seq=ids, mode='forward')

                language_model_loss = compute_loss(outputs, ids, attention_mask)
                output_loss['gen'] = language_model_loss  
            return output_loss
            
        else:

            gen_img_latents, _, _ = self.forward_scan_encoder(imgs, None, mask_ratio=0.0, is_training=is_training)

            gen_img_latents = gen_img_latents[:, 1:, :]  

            global_features = torch.mean(gen_img_latents,dim=1) 

            outputs = self.language_model(fc_feats=global_features, att_feats=gen_img_latents, mode='sample')
            output_ids = outputs[0]   

            return output_ids



    def get_parameter_group(self, lr_ve, lr_ed):
        
        gen_keys = ['language_model']

        msm_parameters = []
        gen_parameters = []
        for name, param in self.named_parameters():
            if any(key in name for key in gen_keys):
                gen_parameters.append(param)
            else:
                msm_parameters.append(param)
        
        return [
            {'params':msm_parameters, 'lr':lr_ve, 'lr_ve':True},
            {'params':gen_parameters, 'lr':lr_ed, 'lr_ed':True}
        ]
