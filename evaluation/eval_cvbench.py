import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="./checkpoints/viral-7b", help='path to the model')
    parser.add_argument('--input_jsonl', type=str, default= "./playground/benchmarks/cvbench/test_2d.jsonl")
    parser.add_argument('--benchmark_dir', type=str, default="./playground/benchmarks/cvbench")
    args = parser.parse_args()

    if 'qwen' in args.model_path:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base="Qwen/Qwen2.5-7B-Instruct",
            model_name="llava-v1.5-7b-qwen2-lora",
        )
        conv_mode = "qwen_2"
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base="lmsys/vicuna-7b-v1.5" if "7b" in args.model_path else "lmsys/vicuna-13b-v1.5",
            model_name="llava-v1.5-7b-lora" if "7b" in args.model_path else "llava-v1.5-13b-lora"
        )
        conv_mode = "llava_v1"

    model.vra_loss = False
    model.residual = False
    model.residual_target_layers = [16]
    json_data = []
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            json_data.append(json.loads(line))

    output_path = f"{os.path.basename(args.model_path)}.json"
    correct = 0
    total = 0
    results = []

    count_total = 0
    count_correct = 0

    relation_total = 0
    relation_correct = 0

    from tqdm import tqdm
    with open(args.output_path, 'a') as f:
        for data in tqdm(json_data):
            image_src = data['filename']
            image = Image.open(os.path.join(args.benchmark_dir, image_src)).convert('RGB')
            question = data['prompt']
            gt_answer = data['answer']
            task = data['task'].lower()
            
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
            conv = conv_templates[conv_mode].copy()

            question += "\nBase your answer on reasoning. Your final answer must be only the single capital letter corresponding to the correct choice."
            prompt = question
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            temperature = 0.1
            max_new_tokens = 10

            model.cache_list = []
            with torch.inference_mode():
                outputs = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  
            model_answer = output_text.strip()

            is_correct = (model_answer.lower() == gt_answer.replace("(", "").replace(")", "").strip().lower())
            correct += int(is_correct)
            total += 1

            if task == "count":
                count_total += 1
                if is_correct:
                    count_correct += 1
            elif task == "relation":
                relation_total += 1
                if is_correct:
                    relation_correct += 1

            f.write(json.dumps({
                "filename": image_src,
                "question": question,
                "gt_answer": gt_answer,
                "model_answer": model_answer,
                "is_correct": is_correct
            }) + '\n')


        accuracy = correct / total * 100
        print(f"{os.path.basename(args.model_path)} || Accuracy: {accuracy:.2f}%")
        print(f"{os.path.basename(args.model_path)} || COUNT :: {count_correct}/{count_total}")
        print(f"{os.path.basename(args.model_path)} || RELATION :: {relation_correct}/{relation_total}")