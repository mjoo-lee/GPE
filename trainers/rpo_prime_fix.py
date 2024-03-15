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
        assert cfg.TRAINER.RPO.K2 >= 1, "K should be bigger than 0"

        # get num of prompts
        self.K1 = cfg.TRAINER.RPO.K1 # the number of prompt pair
        self.K2 = cfg.TRAINER.RPO.K2
        self.K = self.K1 + self.K2
        self.dtype = clip_model.dtype

        # get dim encoders
        self.d_t = clip_model.ln_final.weight.shape[0] #512
        self.d_v = 768

        # check im size
        clip_imsize = clip_model.visual.input_resolution # 224
        cfg_imsize = cfg.INPUT.SIZE[0] # (224, 224)[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # init tokens
        self.initialization_token(clip_model)
        
    def initialization_token(self, clip_model):
        #### text token initialization #####
        self.org_text_token = clip_model.token_embedding(torch.tensor([49407])) #([1,512]), 49407 : vocab_size #cls token EOS 토근을 embedding화
        
        text_token = self.org_text_token.repeat(self.K1, 1) #K개의 prompt길이 만큼 연결 #([24,512]), 18개의 기존 RPO
        text_token += 0.1 * torch.randn_like(text_token)

        prompts = torch.empty(self.K2, self.d_t).normal_(std=0.02)
        prompts = torch.cat([text_token, prompts], dim=0) #([24,512])
        prompts = prompts.type(self.dtype) #데이터타입 변경
        self.text_prompt = nn.Parameter(prompts) #모델에서 학습 가능한 파라미터로 취급

        #### visual token initialization ####
        self.org_visual_token = clip_model.visual.class_embedding #([24,512])

        visual_token = self.org_visual_token.repeat(self.K1, 1) #([24,512])
        visual_token += 0.1 * torch.randn_like(visual_token)

        visual_prompts = torch.empty(self.K2, self.d_v).normal_(std=0.02)
        visual_prompts = torch.cat([visual_token, visual_prompts], dim=0) #([24,512])
        visual_prompts = visual_prompts.type(self.dtype)
        self.img_prompt = nn.Parameter(visual_prompts)

    def forward(self):
        return self.text_prompt, self.img_prompt


class CustomCLIP(nn.Module):
    '''
    cfg : model parameters
    device : model device
    layer : # of query generate FFN layers
    '''
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
        self.text_prompt, self.image_prompt = self.prompt_learner()

        # make prompts for classes
        self.dtype = clipmodel.dtype
        self.prompts = self.make_prompts(classnames, prompt) # ["a photo of a dog.", ".."]
        self.num_classes = len(classnames)

        # define mask
        self.prompt_attention = False
        self.rpo_prime = True
        self.define_mask(prompt_attention=self.prompt_attention, rpo_prime = self.rpo_prime)

        # preprocess text
        # for i in range(self.prompt_learner.K):
        #    self.text_x[torch.arange(self.text_x.shape[0]), self.len_prompts + i, :] = self.text_prompt[i, :].repeat(self.text_x.shape[0], 1)
        #self.text_x += self.text_pos_embedding.type(self.dtype)

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
            text_mask[idx + self.prompt_learner.K1:, :].fill_(float("-inf"))

            # RPO self attention
            if prompt_attention:
                text_mask[idx:(idx+self.prompt_learner.K1), idx:(idx+self.prompt_learner.K1)].fill_diagonal_(0)

            if rpo_prime:
                # rpo-to-extra
                text_mask[idx:(idx+self.prompt_learner.K1), (idx+self.prompt_learner.K1):(idx+self.prompt_learner.K)].fill_(0)

                # extra-to-extra
                text_mask[(idx + self.prompt_learner.K1):(idx + self.prompt_learner.K), (idx + self.prompt_learner.K1):(idx + self.prompt_learner.K)].fill_(0)
            text_mask_list.append(text_mask.repeat(attn_head, 1, 1))  # Repeat for attention heads
        self.text_mask = torch.cat(text_mask_list, dim=0)

        #################image##################
        att_size = 1 + 14 * 14 + self.prompt_learner.K

        # init mask
        visual_mask = torch.zeros((att_size, att_size), dtype=self.dtype, requires_grad=False)

        # cut after input
        visual_mask[:, -self.prompt_learner.K:].fill_(float("-inf"))
        visual_mask[-self.prompt_learner.K:, :].fill_(float("-inf"))

        # set rpo
        visual_mask[-self.prompt_learner.K:(-self.prompt_learner.K+self.prompt_learner.K1), :-self.prompt_learner.K].fill_(0)

        # RPO self attention
        if prompt_attention:
            visual_mask[-self.prompt_learner.K:(-self.prompt_learner.K+self.prompt_learner.K1),-self.prompt_learner.K:(-self.prompt_learner.K+self.prompt_learner.K1)].fill_diagonal_(0)
        
        if rpo_prime:
            visual_mask[-self.prompt_learner.K:(-self.prompt_learner.K+self.prompt_learner.K1), (-self.prompt_learner.K+self.prompt_learner.K1):].fill_(0)
            visual_mask[(-self.prompt_learner.K+self.prompt_learner.K1):,(-self.prompt_learner.K+self.prompt_learner.K1):].fill_(0)

        self.visual_mask = visual_mask
        
    def forward(self, image, label=None):
        device = image.device
        batch_size = image.shape[0]

        # load mask from predefined masks
        K1 = self.cfg.TRAINER.RPO.K1
        K2 = self.cfg.TRAINER.RPO.K2

        ####################### text ###########################        
        text_x = self.text_x.to(device)  # NLD -> LND
        for i in range(K1 + K2):
            text_x[torch.arange(text_x.shape[0]), self.len_prompts + i] = self.text_prompt[i].unsqueeze(0).repeat(text_x.shape[0], 1)
        text_x += self.text_pos_embedding.type(self.dtype)
        text_x = self.text_transformers(text_x.permute(1, 0, 2), self.text_mask.to(device))
        text_x = text_x.permute(1, 0, 2)
        input(text_x)
        text_x = self.text_ln_final(text_x).type(self.dtype) @ self.text_proj
        input(text_x)
        text_rpo_f = []
        for i in range(K1 if self.rpo_prime else K1 + K2):
            idx = self.len_prompts + i
            text_rpo_f.append(text_x[torch.arange(text_x.shape[0]), idx].unsqueeze(1))
        text_rpo_f = torch.cat(text_rpo_f, dim=1)
        text_org_f = text_x[torch.arange(text_x.shape[0]), self.text_tokenized.argmax(dim=-1)]
        text_rpo_f = text_rpo_f / text_rpo_f.norm(dim=-1, keepdim=True) #[C, K, D]
        text_org_f = text_org_f / text_org_f.norm(dim=-1, keepdim=True) #[C, D]
        text_all_f = torch.cat([text_rpo_f, text_org_f.unsqueeze(1)], dim=1) #[C, K+1, D]
        
        ####################### img ###########################
        image_embedding = self.img_patch_embedding(image.type(self.dtype)) # (batch_size, h_dim, 7, 7)
        image_embedding = image_embedding.reshape(batch_size, image_embedding.shape[1], -1)
        image_embedding = image_embedding.permute(0,2,1) # (batch_size, 49, h_dim)
        image_embedding = torch.cat([self.img_cls_embedding.repeat(batch_size,1,1).type(self.dtype), image_embedding], dim=1) # 16 (batch_size, 50, h_dim)
        img_x = torch.cat([image_embedding, self.image_prompt.repeat(batch_size, 1, 1)], dim=1) #(batch, K, dim)
        img_x += torch.cat([self.img_pos_embedding, self.img_pos_embedding[0].unsqueeze(0).repeat(K1 + K2, 1)], 0).type(self.dtype)
        img_x = self.img_pre_ln(img_x)
        img_x = img_x.permute(1, 0, 2)
        img_x = self.img_transformer(img_x, self.visual_mask.to(device))
        img_x = img_x.permute(1, 0, 2)
        img_f = self.img_post_ln(img_x) @ self.img_proj
        if self.rpo_prime:
            img_rpo_f = img_f[:, -1 * self.prompt_learner.K: -1 * (self.prompt_learner.K-K1)]
        else:
            img_rpo_f = img_f[:, -1 * self.prompt_learner.K:]
        img_org_f = img_f[:, 0]
        img_rpo_f = img_rpo_f / img_rpo_f.norm(dim=-1, keepdim=True) # [B, K, D]
        img_org_f = img_org_f / img_org_f.norm(dim=-1, keepdim=True) # [B, D]
        img_all_f = torch.cat([img_rpo_f, img_org_f.unsqueeze(1)], dim=1) # [B, K+1, D]

        print(text_all_f, img_all_f)
        input()
        ####################### logit ###########################
        zs_logit = self.logit_scale.exp() * img_org_f @ text_org_f.t() #[B, C]
        mix_logit = self.logit_scale.exp() * torch.einsum('pbe,Pec->pPbc',img_all_f.permute(1, 0, 2), text_all_f.permute(1,2,0))
        mix_logit = mix_logit.reshape(-1, batch_size, self.num_classes).to(device)
        print(mix_logit.shape, label.shape)
        print(mix_logit)
        if self.prompt_learner.training:
            num_models = mix_logit.shape[0]

            # ce loss
            ce_loss = F.cross_entropy(mix_logit.reshape([-1, self.num_classes]),
                                      label.reshape([1, -1]).repeat([num_models, 1]).reshape([-1, 1]).squeeze(),
                                      reduction='none', label_smoothing=0.2)
            ce_loss = ce_loss.reshape([num_models, batch_size]) # (num_models, num_samples)

            masking = (torch.rand_like(ce_loss) > 0.5).to(device)
            loss = ce_loss * masking
            print(torch.sum(ce_loss))
            input()
            return torch.sum(loss) / torch.sum(masking)
        
        #동근 코드
        # if self.prompt_learner.training:
        #     #### beta regularization ##########
        #     beta = 0.1
        #     num_models = logits_324.shape[0]
        #     num_samples = logits_324.shape[1]
        #     num_classes = logits_324.shape[2]
        #     # ce loss
        #     ce_loss = F.cross_entropy(logits_324.reshape([-1, num_classes]),
        #                             label.reshape([1, -1]).repeat([num_models, 1]).reshape([-1, 1]).squeeze(),
        #                             reduction='none', label_smoothing=0.0)
        #     ce_loss = ce_loss.reshape([num_models, -1]) # (num_models, num_samples)
        #     # pdb.set_trace()
        #     # regularizer
        #     reg_loss = F.cross_entropy(logits_324.reshape([-1, num_classes]),
        #                             torch.ones(num_models*num_samples, num_classes).to(device)/num_classes,
        #                             reduction='none', label_smoothing=0.0)
        #     reg_loss = beta * reg_loss.reshape([num_models, -1])  # (num_models, num_samples)
        #     #  masking and combine (교수님)
        #     masking = torch.eye(num_models).unsqueeze(-1).to(device)
        #     tmp_loss = masking * ce_loss.unsqueeze(0).repeat(num_models, 1, 1) + (1 - masking) * reg_loss.unsqueeze(0).repeat(num_models, 1, 1)
        #     tmp_loss = tmp_loss.sum(0) # (num_models, num_samples)
        #     # model selection
        #     masking = torch.zeros(num_models, num_samples).to(device)
        #     idx = torch.argsort( tmp_loss, descending=False,dim=0) # torch.Size([384, 4])
        #     idx = idx[:int(num_models*0.7)]                     # torch.Size([162, 4])
        #     for i in range(num_samples):
        #         masking[idx[:, i], i] = 1
        #     # masking[tmp_loss.min(0)[-1], torch.arange(num_samples)] = 1
        #     # compute final
        #     final_loss = masking * ce_loss  + (1 - masking) * reg_loss
        #     return final_loss.mean()
        
        #####
        #inference -> ensemble
        return F.softmax(mix_logit, -1).mean(0)

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












  
