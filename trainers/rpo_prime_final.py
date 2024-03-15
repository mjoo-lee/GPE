## RPO 


import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import pdb

_tokenizer = _Tokenizer()
import pdb

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        # Make sure K >= 1
        assert cfg.TRAINER.RPO.K1 >= 1, "K should be bigger than 0"
        #assert cfg.TRAINER.RPO.K2 >= 1, "K should be bigger than 0"

        # get num of prompts
        self.K1 = cfg.TRAINER.RPO.K1
        #self.K2 = cfg.TRAINER.RPO.K2
        self.K = self.K1 #+ self.K2
        self.dtype = clip_model.dtype

        # get dim encoders
        self.d_t = clip_model.ln_final.weight.shape[0]
        self.d_v = 768

        # check im size
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # init tokens
        self.initialization_token(clip_model)

    def initialization_token(self, clip_model):
        text_token = clip_model.token_embedding(torch.tensor([49407])).repeat(self.K1, 1)
        text_token = text_token + 0.1 * torch.randn_like(text_token)

        # prompts = torch.randn(self.K2, self.d_t) * 0.02
        # prompts = torch.cat([text_token, prompts], dim=0)
        prompts = text_token #12/12/0 세팅 추가
        self.text_prompt = nn.Parameter(prompts.type(self.dtype))

        visual_token = clip_model.visual.class_embedding.repeat(self.K1, 1)
        visual_token = visual_token + 0.1 * torch.randn_like(visual_token)

        # visual_prompts = torch.randn(self.K2, self.d_v) * 0.02
        # visual_prompts = torch.cat([visual_token, visual_prompts], dim=0)
        visual_prompts = visual_token #12/12/0 세팅 추가
        self.img_prompt = nn.Parameter(visual_prompts.type(self.dtype))

    def forward(self):
        return self.text_prompt, self.img_prompt


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, prompt, clipmodel):
        super().__init__()
        self.cfg = cfg

        # text encoder
        self.token_embedding = clipmodel.token_embedding
        self.text_pos_embedding = clipmodel.positional_embedding
        self.text_transformers = clipmodel.transformer
        self.text_ln_final = clipmodel.ln_final
        self.text_proj = clipmodel.text_projection

        # vision encoder
        self.img_patch_embedding = clipmodel.visual.conv1
        self.img_cls_embedding = clipmodel.visual.class_embedding
        self.img_pos_embedding = clipmodel.visual.positional_embedding
        self.img_pre_ln = clipmodel.visual.ln_pre
        self.img_transformer = clipmodel.visual.transformer
        self.img_post_ln = clipmodel.visual.ln_post
        self.img_proj = clipmodel.visual.proj

        # logit
        self.logit_scale = clipmodel.logit_scale

        # initialization token
        self.prompt_learner = PromptLearner(self.cfg, clipmodel)
        self.K = self.prompt_learner.K
        self.K1 = self.prompt_learner.K1
        #self.K2 = self.prompt_learner.K2
        # self.num_models = (self.K1 + 1) * (self.K1 + 1)
        self.num_models = (self.K1 ) * (self.K1 )
        # make prompts for classes
        self.dtype = clipmodel.dtype
        self.prompts = self.make_prompts(classnames, prompt)
        self.num_classes = len(classnames)

        # define mask
        self.prompt_attention = True
        self.rpo_prime = True
        self.define_mask(prompt_attention=self.prompt_attention, rpo_prime=self.rpo_prime)

    def make_prompts(self, classnames, prompt):
        prompts = [prompt.replace('_', c) for c in classnames]
        with torch.no_grad():
            self.text_tokenized = torch.cat([clip.tokenize(p) for p in prompts])
            self.text_x = self.token_embedding(self.text_tokenized).type(self.dtype)
            self.len_prompts = self.text_tokenized.argmax(dim=-1) + 1
        return prompts

    def define_mask(self, prompt_attention=False, rpo_prime=False):
        len_max = 77
        attn_head = 8

        #################text##################
        # build mask for each class
        text_mask_list = []
        for idx in self.len_prompts:
            # init causal mask
            text_mask = torch.full((len_max, len_max), float("-inf"), dtype=torch.float32, requires_grad=False)
            text_mask = torch.triu(text_mask, diagonal=1)

            # cut after input length
            text_mask[:, idx:].fill_(float("-inf"))

            # cut after prompt
            text_mask[idx + self.K1:idx + self.K, :].fill_(float("-inf"))

            # RPO self attention
            if prompt_attention:
                text_mask[idx:(idx + self.K1), idx:(idx + self.K1)].fill_diagonal_(0)

            if rpo_prime:
                # rpo-to-extra
                text_mask[(idx + self.K1 // 2):(idx + self.K1), (idx + self.K1):(idx + self.K)].fill_(0)

                # extra-to-extra
                text_mask[(idx + self.K1):(idx + self.K), (idx + self.K1):(idx + self.K)].fill_(0)
            text_mask_list.append(text_mask.repeat(attn_head, 1, 1))  # Repeat for attention heads
        self.text_mask = torch.cat(text_mask_list, dim=0)

        #################image##################
        att_size = 1 + 14 * 14 + self.K

        # init mask
        visual_mask = torch.zeros((att_size, att_size), dtype=self.dtype, requires_grad=False)

        # cut after input
        visual_mask[:, -self.K:].fill_(float("-inf"))
        visual_mask[-self.K:, :].fill_(float("-inf"))

        # set rpo
        visual_mask[-self.K:(-self.K + self.K1), :-self.K].fill_(0)

        # RPO self attention
        if prompt_attention:
            visual_mask[-self.K:(-self.K + self.K1), -self.K:(-self.K + self.K1)].fill_diagonal_(0)

        if rpo_prime:
            visual_mask[(-self.K + self.K1 // 2):(-self.K + self.K1), (-self.K + self.K1):].fill_(0)
            visual_mask[(-self.K + self.K1):, (-self.K + self.K1):].fill_(0)

        self.visual_mask = visual_mask

    def forward(self, image, label=None):
        device = image.device
        batch_size = image.shape[0]
        text_prompt, image_prompt = self.prompt_learner()

        ####################### text ###########################        
        text_x = self.text_x
        text_x = text_x.to(device)
        for i in range(self.K):
            text_x[torch.arange(text_x.shape[0]), self.len_prompts + i] = text_prompt[i].unsqueeze(0).repeat(
                text_x.shape[0], 1)
        text_x += self.text_pos_embedding.type(self.dtype)
        text_x = self.text_transformers(text_x.permute(1, 0, 2), self.text_mask)
        text_x = text_x.permute(1, 0, 2)
        text_x = self.text_ln_final(text_x) @ self.text_proj
        text_rpo_f = []
        for i in range(self.K1 if self.rpo_prime else self.K):
            idx = self.len_prompts + i
            text_rpo_f.append(text_x[torch.arange(text_x.shape[0]), idx].unsqueeze(1))
        text_rpo_f = torch.cat(text_rpo_f, dim=1)
        text_org_f = text_x[torch.arange(text_x.shape[0]), self.text_tokenized.argmax(dim=-1)]
        text_rpo_f = text_rpo_f / text_rpo_f.norm(dim=-1, keepdim=True)
        text_org_f = text_org_f / text_org_f.norm(dim=-1, keepdim=True)
        text_all_f = torch.cat([text_rpo_f, text_org_f.unsqueeze(1)], dim=1)

        ####################### img ###########################
        image_embedding = self.img_patch_embedding(image.type(self.dtype))
        image_embedding = image_embedding.reshape(batch_size, image_embedding.shape[1], -1)
        image_embedding = image_embedding.permute(0, 2, 1)
        image_embedding = torch.cat([self.img_cls_embedding.repeat(batch_size, 1, 1).type(self.dtype), image_embedding],
                                    dim=1)
        img_x = torch.cat([image_embedding, image_prompt.repeat(batch_size, 1, 1)], dim=1)
        img_x += torch.cat([self.img_pos_embedding, self.img_pos_embedding[0].unsqueeze(0).repeat(self.K, 1)], 0).type(
            self.dtype)
        img_x = self.img_pre_ln(img_x)
        img_x = img_x.permute(1, 0, 2)
        img_x = self.img_transformer(img_x, self.visual_mask)
        img_x = img_x.permute(1, 0, 2)
        img_f = self.img_post_ln(img_x) @ self.img_proj
        if self.rpo_prime:
            #img_rpo_f = img_f[:, -1 * self.K: -1 * (self.K - self.K1)]
            img_rpo_f = img_f[:, -1 * self.K:]
        else:
            img_rpo_f = img_f[:, -1 * self.K:]
        img_org_f = img_f[:, 0]
        img_rpo_f = img_rpo_f / img_rpo_f.norm(dim=-1, keepdim=True)
        img_org_f = img_org_f / img_org_f.norm(dim=-1, keepdim=True)
        img_all_f = torch.cat([img_rpo_f, img_org_f.unsqueeze(1)], dim=1)

        ####################### logit ###########################
        mix_logit = self.logit_scale.exp() * torch.einsum('pbe,Pec->pPbc',img_all_f.permute(1, 0, 2), text_all_f.permute(1,2,0))
        mix_logit = mix_logit.reshape(-1, batch_size, self.num_classes).to(device)
    
        first_half_logit = self.logit_scale.exp() * torch.einsum('pbe,Pec->pPbc',img_rpo_f[:, :self.K1//2].permute(1, 0, 2), text_rpo_f[:, :self.K1//2].permute(1,2,0))
        # first_half_logit = self.logit_scale.exp() * torch.einsum('pbe,Pec->pPbc',cls_first_img_rpo_f.permute(1, 0, 2), cls_first_text_rpo_f.permute(1,2,0))
        first_half_logit = first_half_logit.reshape(-1, batch_size, self.num_classes).to(device)
        second_half_logit = self.logit_scale.exp() * torch.einsum('pbe,Pec->pPbc',img_rpo_f[:, self.K1//2:].permute(1, 0, 2), text_rpo_f[:, self.K1//2:].permute(1,2,0))
        # second_half_logit = self.logit_scale.exp() * torch.einsum('pbe,Pec->pPbc',cls_second_img_rpo_f.permute(1, 0, 2), cls_second_text_rpo_f.permute(1,2,0))
        second_half_logit = second_half_logit.reshape(-1, batch_size, self.num_classes).to(device)
        # mix_logit = mix_logit.reshape(-1, batch_size, self.num_classes).to(device)
        if self.prompt_learner.training:
           
            ce_loss = F.cross_entropy(first_half_logit.mean(0), label) + F.cross_entropy(second_half_logit.mean(0), label)
            def get_off_diagonal_elements(M):
                return M[:, ~torch.eye(*M.shape[1:],dtype = torch.bool)]
            def cov_loss(M):
                M = M - M.mean(1).unsqueeze(1)
                M_cov = torch.bmm(M, M.permute(0, 2, 1)) # B x K x K
                M_cov /= (M.shape[1] - 1)
                return get_off_diagonal_elements(M_cov).pow_(2).mean()
            # rpo_f : (B, K, C)
            # first_img_cov_loss = cov_loss(img_rpo_f[:, :self.K1//2])
            # fist_text_cov_loss = cov_loss(text_rpo_f[:, :self.K1//2])
            # second_img_cov_loss = cov_loss(img_rpo_f[:, self.K1//2:])
            # second_text_cov_loss = cov_loss(text_rpo_f[:, self.K1//2:])
            # cov_loss = first_img_cov_loss + fist_text_cov_loss + second_img_cov_loss + second_text_cov_loss
            
            # group1,2 함께 cov loss
            img_cov_loss = cov_loss(img_rpo_f)
            text_cov_loss = cov_loss(text_rpo_f)
            cov_loss = img_cov_loss + text_cov_loss
            #import pdb;pdb.set_trace()
            return ce_loss +  self.cfg.TRAINER.RPO.cov_loss * cov_loss
           

        #####
        # inference -> ensemble
        # mix_logit_softmax = F.softmax(mix_logit, -1)
        # score = mix_logit_softmax.max(2).values.softmax(0)
        # median = score.median(0).values
        # index = (score - median).abs()/median <0.5
        # return (mix_logit*mix_logit_softmax.max(2).values.softmax(0).unsqueeze(2)).mean(0)
        return F.softmax(mix_logit, -1).mean(0)
        # return F.softmax(first_half_logit.mean(0), -1) + F.softmax(second_half_logit.mean(0), -1)


@TRAINER_REGISTRY.register()
class RPO_prime(TrainerX):
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.RPO.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.RPO.PREC == "fp32" or cfg.TRAINER.RPO.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        prompt = cfg.DATASET.PROMPT
        ############################################# 통일 #####

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, prompt, clip_model)

        # parameter freeze
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.RPO.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        # nan detector
        torch.autograd.set_detect_anomaly(True)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.RPO.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
















