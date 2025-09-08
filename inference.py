import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/viral-7b")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    return parser.parse_args()

args = parse_args()
disable_torch_init()

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=args.model_path,
    model_base="lmsys/vicuna-7b-v1.5" if "7b" in args.model_path else "lmsys/vicuna-13b-v1.5",
    model_name="llava-v1.5-7b-lora" if "7b" in args.model_path else "llava-v1.5-13b-lora"
)

model.vra_loss = False
model.residual = False
model.residual_target_layers = [16]

image_path = "./images/llava_logo.png"
image = Image.open(image_path).convert('RGB')
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

model_name = os.path.basename(args.model_path)
conv_mode = "llava_v1" 
conv = conv_templates[conv_mode].copy()

prompt = "Is the lizard facing left or right from the camera's perspective?"
inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

with torch.inference_mode():
    outputs = model.generate(
        inputs=input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_attentions=True,
    )

img_token_where = (input_ids == IMAGE_TOKEN_INDEX)
output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
hidden_states = outputs.hidden_states

tokens_to_insert = 24 * 24 
new_masks = []
for b in range(img_token_where.size(0)):
    seq_len = input_ids.size(1)
    orig_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
    true_indices = torch.nonzero(img_token_where[b], as_tuple=False).flatten()

    new_mask = []
    curr_idx = 0
    for idx in true_indices:
        idx = idx.item()
        new_mask.append(orig_mask[curr_idx:idx])
        new_mask.append(torch.ones(tokens_to_insert, dtype=torch.bool, device=input_ids.device))
        curr_idx = idx + 1 

    # Append remaining part
    new_mask.append(orig_mask[curr_idx:])

    # Concat full mask
    new_mask = torch.cat(new_mask, dim=0)
    new_masks.append(new_mask)

new_masks = torch.stack(new_masks, dim=0)
target_layer = hidden_states[0][16]
target_feature = target_layer[:, new_masks[0], :].squeeze(0)

def visualize_feature_rgb_improved():
    features_np = target_feature.detach().cpu().numpy()
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(features_np)
    median = np.median(feat_pca, axis=0)
    q1 = np.percentile(feat_pca, 25, axis=0)
    q3 = np.percentile(feat_pca, 75, axis=0)
    iqr = q3 - q1
    scaled = (feat_pca - median) / (iqr + 1e-6)
    feat_pca_norm = 0.5 * (np.tanh(scaled) + 1)
    rgb_image = feat_pca_norm.reshape(24, 24, 3)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title('Feature Map Visualization (24x24)')
    plt.axis('off')  
    plt.savefig(f'feature_rgb_map_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rgb_image

rgb_image = visualize_feature_rgb_improved()

if conv.sep_style == SeparatorStyle.TWO:
    response = output_text.split(conv.sep2)[-1].strip()
elif conv.sep_style == SeparatorStyle.LLAMA_2:
    response = output_text.split(conv.roles[1] + ":")[-1].strip()
else:
    response = output_text.split(conv.roles[1] + ":")[-1].strip()

print("Input:", prompt)
print("\nResponse:", response)