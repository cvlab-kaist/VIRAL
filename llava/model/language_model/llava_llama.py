#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import wandb

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# IMAGENET_DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
# IMAGENET_DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711]

# CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
# CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def visualize_feature_pca(features, save_path='feature.png'):
    # CPU로 이동하고 numpy로 변환
    features_np = features.float().squeeze(0).detach().cpu().numpy()
    
    # PCA로 768차원을 3차원으로 축소
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(features_np)
    
    # 정규화: 중앙값과 IQR 기반
    median = np.median(feat_pca, axis=0)
    q1 = np.percentile(feat_pca, 25, axis=0)
    q3 = np.percentile(feat_pca, 75, axis=0)
    iqr = q3 - q1
    scaled = (feat_pca - median) / (iqr + 1e-6)
    feat_pca_norm = 0.5 * (np.tanh(scaled) + 1)
    
    # 24x24x3 이미지로 재구성
    size = int(np.sqrt(features_np.shape[0]))  # target_size//16
    rgb_image = feat_pca_norm.reshape(size, size, 3)
    
    # 시각화 및 저장
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title(f'Feature Map Visualization ({size}x{size})')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return rgb_image


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def is_main_process():
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
        return is_main_process
    return True

class AlignmentProjector(nn.Module):
    def __init__(self, hidden_size, projector_dim, z_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dim),
        )

    def forward(self, x):
        return self.projector(x)

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

class ResidualLlamaModel(LlamaModel):
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        residual: Optional[bool] = False,
        target_layers: Optional[List[int]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if residual:
            assert target_layers is not None, "target_layers must be specified if residual is True"
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if residual and (idx+1 in target_layers):
                hidden_states = hidden_states + inputs_embeds

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlavaLlamaModel(LlavaMetaModel, ResidualLlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vra_loss = config.vra_loss if hasattr(config, 'vra_loss') else False
        self.residual = config.residual if hasattr(config, 'residual') else False
        self.diffusion_loss = config.diffusion_loss if hasattr(config, 'diffusion_loss') else False
        if self.diffusion_loss:
            self.diffusion_weight = config.diffusion_weight if hasattr(config, 'diffusion_weight') else 1.0
        assert not(self.vra_loss & self.residual), "vra loss and residual cannot be true at same time"
        self.residual_target_layers = config.residual_target_layers if hasattr(config, 'residual_target_layers') else None
        if self.vra_loss:
            self.only_coco = config.only_coco if hasattr(config, 'only_coco') else False
            self.target_layers = config.target_layers if hasattr(config, 'target_layers') else [15,16]
            self.vra_target = config.vra_target if hasattr(config, 'vra_target') else "dinov2-vit-b" # sam_vit_b_01ec64, dinov2-vit-b, clip
            self.vra_weight = config.vra_weight if hasattr(config, 'vra_weight') else 0.5
            self.projector_dim = config.projector_dim if hasattr(config, 'projector_dim') else 2048 # VRA Default
            self.z_dim = config.z_dim if hasattr(config, 'z_dim') else 768 # DINO Default 768, 256 if SAM, 1024 if CLIP-L
            self.alignment_loss = config.alignment_loss if hasattr(config, 'alignment_loss') else "direct" #"direct" # direct, similarity
            self.use_projector = config.use_projector if hasattr(config, 'use_projector') else False # If False, use the mid hidden states directly, only for similarity loss.
            self.use_multiple_projectors = config.use_multiple_projectors if hasattr(config, 'use_multiple_projectors') else False # If True, use multiple projectors for each layer, only for similarity loss.
            if self.target_layers is not None:
                if self.use_multiple_projectors:
                    self.alignment_projector = nn.ModuleList([
                        AlignmentProjector(config.hidden_size, self.projector_dim, self.z_dim) for _ in range(len(self.target_layers))
                    ])
                else:
                    self.alignment_projector = AlignmentProjector(config.hidden_size, self.projector_dim, self.z_dim)
                if 'dinov2' in self.vra_target:
                    import timm
                    model, _, model_config = self.vra_target.split('-')
                    if 'reg' in self.vra_target:
                        self.alignment_encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
                    else:
                        self.alignment_encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
                    del self.alignment_encoder.head
                    patch_resolution = 24
                    self.alignment_encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                        self.alignment_encoder.pos_embed.data, [patch_resolution, patch_resolution],
                    )
                    self.alignment_encoder.head = nn.Identity()
                elif 'sam' in self.vra_target:
                    from segment_anything import sam_model_registry, SamPredictor
                    if os.path.exists(self.vra_target):
                        sam_checkpoint = self.vra_target
                    else:
                        sam_checkpoint = f"./playground/vfm_weights/{self.vra_target}.pth"
                    model_type = "vit_b" if "vit_b" in sam_checkpoint else "vit_l" if "vit_l" in sam_checkpoint else "vit_h"
                    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                    predictor = SamPredictor(sam)
                    self.alignment_encoder = predictor.model.image_encoder
                    del sam, predictor
                    torch.cuda.empty_cache()
                    patch_resolution = 24
                elif 'clip' in self.vra_target:
                    self.alignment_encoder = None
                    # we will use CLIPVisionTower later on
                elif 'radio' in self.vra_target.lower():
                    self.alignment_encoder = torch.hub.load('NVlabs/RADIO', 'radio_model', version=self.vra_target, progress=True, skip_validation=True)
                elif 'depth_anything' in self.vra_target.lower():
                    from ..depth_anything_v2.dpt import DepthAnythingV2
                    
                    model_configs = {
                        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                    }
                    
                    if "vitb" in self.vra_target.lower():
                        encoder = "vitb"
                    else:
                        raise NotImplementedError(f"Unknown encoder type for Depth-Anything: {self.vra_target}")
                    
                    dv2 = DepthAnythingV2(**model_configs[encoder])
                    dv2.load_state_dict(torch.load(f'./playground/vfm_weights/{self.vra_target}.pth', map_location='cpu'))
                    self.alignment_encoder = dv2.pretrained
                    
                    del dv2
                    torch.cuda.empty_cache()
                    
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward_dep(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None, # Need input ids for vra loss
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        is_coco = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        
        img_token_where = (input_ids == -200)
        if self.training and inputs_embeds is not None:
            input_ids = None
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.vra_loss:
            output_hidden_states = True
            
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            residual=self.residual,
            target_layers=self.residual_target_layers
        )

        hidden_states = outputs[0]
        if self.vra_loss:
            if self.target_layers is None:
                self.target_layers = list(range(0, len(self.model.layers)+1))

            mid_hidden_states = [outputs.hidden_states[i] for i in self.target_layers]
            
            del outputs.hidden_states
            
            if 'dinov2' in self.vra_target:
                images_resized = F.interpolate(images, size=(336, 336), mode="bilinear") # Maybe it is better to use 224 and upsample feature or downsample llm feature
                self.alignment_encoder.eval()
                with torch.no_grad():
                    alignment_feature = self.alignment_encoder.forward_features(images_resized)['x_norm_patchtokens']
                del images_resized
            elif 'clip' in self.vra_target:
                self.alignment_encoder = self.model.vision_tower # Assume grad in disabled
                assert self.alignment_encoder is not None, "CLIP vision tower is not loaded."
                images_resized = F.interpolate(images, size=(336, 336), mode="bilinear") # Maybe it is better to use 224 and upsample feature or downsample llm feature
                self.alignment_encoder.eval()
                with torch.no_grad():
                    alignment_feature = self.alignment_encoder(images_resized)
                del images_resized
            elif 'sam' in self.vra_target:
                images_resized = F.interpolate(images, size=(384, 384), mode="bilinear")
                padded_size = 1024
                feature_size = 24
                normalized_mean = torch.tensor([0, 0, 0], dtype=images_resized.dtype, device=images_resized.device).view(1, 3, 1, 1)
                padded_images = torch.ones((images_resized.shape[0], 3, padded_size, padded_size), dtype=images_resized.dtype, device=images_resized.device) * normalized_mean
                start_h = (padded_size - 384) // 2
                start_w = (padded_size - 384) // 2
                padded_images[:, :, start_h:start_h+384, start_w:start_w+384] = images_resized
            
                self.alignment_encoder.eval()
                with torch.no_grad():
                    alignment_feature = self.alignment_encoder(padded_images)
                del images_resized, padded_images
                B, C, H, W = alignment_feature.shape
                start_idx = (H - feature_size) // 2
                end_idx = start_idx + feature_size
                alignment_feature = alignment_feature[:, :, start_idx:end_idx, start_idx:end_idx] # [B, C, 24, 24]
                alignment_feature = alignment_feature.permute(0, 2, 3, 1).reshape(B, -1, C) # [B, 576, C]
            elif 'radio' in self.vra_target.lower():
                images_resized = F.interpolate(images, size=(384, 384), mode="bilinear")
                
                mean = torch.tensor(self.model.vision_tower.image_mean, dtype=images_resized.dtype, device=images_resized.device).view(1, 3, 1, 1)
                std = torch.tensor(self.model.vision_tower.image_std, dtype=images_resized.dtype, device=images_resized.device).view(1, 3, 1, 1)
                images_resized = torch.clamp(images_resized * std + mean, 0, 1)
                
                self.alignment_encoder.eval()
                with torch.no_grad():
                    if images_resized.dtype == torch.bfloat16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            summary, alignment_feature = self.alignment_encoder(images_resized)
                    else:
                        summary, alignment_feature = self.alignment_encoder(images_resized)
                del images_resized, summary
            elif 'depth_anything' in self.vra_target:
                images_resized = F.interpolate(images, size=(336, 336), mode="bilinear") # Maybe it is better to use 224 and upsample feature or downsample llm feature
                if 'vitb' in self.vra_target.lower():
                    target_layer = 11
                else:
                    raise NotImplementedError(f"Unknown Depth-Anything model: {self.vra_target}")
                
                self.alignment_encoder.eval()
                with torch.no_grad():
                    alignment_feature = self.alignment_encoder.get_intermediate_layers(images_resized, [target_layer], return_class_token=False)[0]
                del images_resized
            
              
            mid_hidden_states = torch.stack(mid_hidden_states, dim=1)
            bsz, num_layers, seq_len, hidden_size = mid_hidden_states.shape # seq_len should be change to image patches
            if self.alignment_loss == "direct" or (self.alignment_loss == "similarity" and self.use_projector):
                if self.use_multiple_projectors:
                    projected_feature = []
                    for idx in range(len(self.target_layers)):
                        proj_b = self.alignment_projector[idx](mid_hidden_states[:, idx, :])
                        projected_feature.append(proj_b)
                    projected_feature = torch.stack(projected_feature, dim=1)
                else:
                    mid_hidden_states = mid_hidden_states.view(-1, seq_len, hidden_size) # flatten the layers
                    projected_feature = self.alignment_projector(mid_hidden_states)
                    projected_feature = projected_feature.view(bsz, num_layers, seq_len, -1) # reshape to bsz, num_layers, seq_len, z_dim
            elif self.alignment_loss == "similarity":
                projected_feature = mid_hidden_states
            
            del mid_hidden_states
            torch.cuda.empty_cache()
            
            vra_loss = 0.
            valid_batch_count = 0
            for b in range(bsz):
                img_tokens_b = img_token_where[b]
                if img_tokens_b.any() and img_tokens_b.sum() == 576:
                    if self.only_coco and not is_coco[b]:
                        continue
                    valid_batch_count += 1
                    for idx in range(len(self.target_layers)):
                        if self.alignment_loss == "direct":
                            proj_b = projected_feature[b, idx, img_tokens_b, :]
                            proj_b = F.normalize(proj_b, dim=-1)
                            alig_b = F.normalize(alignment_feature[b], dim=-1) # normalize the alignment feature
                            vra_loss += (-(proj_b * alig_b).sum(dim=-1)).mean()
                        elif self.alignment_loss == "similarity":
                            proj_b = projected_feature[b, idx, img_tokens_b, :]
                            alig_b = alignment_feature[b]
                            proj_b = F.normalize(proj_b, dim=-1)
                            alig_b = F.normalize(alig_b, dim=-1)
                            
                            proj_b = torch.matmul(proj_b, proj_b.transpose(-2, -1))
                            alig_b = torch.matmul(alig_b, alig_b.transpose(-2, -1))
                            sim_loss = F.mse_loss(proj_b, alig_b)
                            vra_loss += sim_loss
                        else:
                            raise ValueError(f"Unknown alignment loss: {self.alignment_loss}")
            if valid_batch_count > 0:
                vra_loss /= (valid_batch_count * len(self.target_layers))
            
            
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if self.vra_loss:
            print(f"VRA loss: {vra_loss} || NTP loss: {loss}")
            if is_main_process() and self.training:
                wandb.log({"ntp loss": loss.item()})
                if vra_loss == 0.0:
                    wandb.log({"vra loss": 0.0})
                else:
                    wandb.log({"vra loss": vra_loss.item()})
            loss = loss + self.vra_weight * vra_loss
        elif self.residual:
            if self.training:
                print(f"VRA loss: {0.0} || NTP loss: {loss}")
                if is_main_process():
                    wandb.log({"ntp loss": loss.item()})
        
        diffusion_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.diffusion_loss:
            bsz = hidden_states.shape[0]
            if img_token_where.sum() == bsz * 576:
                diffusion_loss = self.compute_vm_loss(images, hidden_states, img_token_where)
                
            print(f"Diffusion loss: {diffusion_loss} || NTP loss: {loss}")
            if is_main_process():
                wandb.log({"diffusion loss": diffusion_loss.item()})
                wandb.log({"ntp loss": loss.item()})
            loss = loss + self.diffusion_weight * diffusion_loss
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
