import os
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import re
from collections import defaultdict

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='SEED-Bench Evaluation for LLaVA')
    parser.add_argument("--model-path", type=str, default="./checkpoints/viral-7b")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    return parser.parse_args()

def extract_answer(text):
    """Extract the answer (A, B, C, or D) from the model's response."""
    text = text.lower()
    
    # Check for direct mention of A, B, C, D
    if re.search(r'\b(a|option a)\b', text):
        return "A"
    elif re.search(r'\b(b|option b)\b', text):
        return "B"
    elif re.search(r'\b(c|option c)\b', text):
        return "C"
    elif re.search(r'\b(d|option d)\b', text):
        return "D"
    
    # Default return if nothing found
    print(f"Warning: Could not extract answer from: '{text}'")
    return "unknown"

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    disable_torch_init()
    
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
    
    
    model.vra_loss = False  # VIRAL loss 비활성화
    # Set model to evaluation mode
    model.eval()
    
    dataset = load_dataset("Lin-Chen/MMStar", "val")
    # breakpoint()
    
    # Prepare for evaluation
    results = []
    category_counts = defaultdict(int)
    category_correct = defaultdict(int)
    
    l2_category_counts = defaultdict(int)
    l2_category_correct = defaultdict(int)
    
    # Evaluation loop
    for qa_item in tqdm(dataset["val"]):
        category = qa_item['category']
        l2_category = qa_item['l2_category']
        image = qa_item['image']
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        question = qa_item['question']
        question += "\nBase your answer on reasoning. Your final answer must be only the single capital letter corresponding to the correct choice."
 
        # Set up conversation
        conv = conv_templates[conv_mode].copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # Run inference
        with torch.inference_mode():
            outputs = model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        
        # Decode output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response
        if conv.sep_style == SeparatorStyle.TWO:
            response = output_text.split(conv.sep2)[-1].strip()
        elif conv.sep_style == SeparatorStyle.LLAMA_2:
            response = output_text.split(conv.roles[1] + ":")[-1].strip()
        else:
            response = output_text.split(conv.roles[1] + ":")[-1].strip()
        
        pred = extract_answer(response)
        gt = qa_item['answer']
        is_correct = (pred == gt)
        
        if is_correct:
            category_correct[category] += 1
            l2_category_correct[l2_category] += 1
        category_counts[category] += 1
        l2_category_counts[l2_category] += 1
        
        # Save result
        results.append({
            "question_id": qa_item['index'],
            "category": qa_item['category'],
            "l2_category": qa_item['l2_category'],
            "question": qa_item['question'],
            "prediction": pred,
            "ground_truth": gt,
            "correct": is_correct,
            "full_response": response
        })
    
    # Calculate accuracy
    total_correct = sum(category_correct.values())
    total_count = sum(category_counts.values())
    overall_accuracy = total_correct / total_count if total_count > 0 else 0
    
    for category in sorted(category_counts.keys()):
        category_accuracy = category_correct[category] / category_counts[category] if category_counts[category] > 0 else 0
        print(f"{category}: {category_accuracy:.4f} ({category_correct[category]}/{category_counts[category]})")
    
    print("-" * 50)
    for l2_category in sorted(l2_category_counts.keys()):
        l2_category_accuracy = l2_category_correct[l2_category] / l2_category_counts[l2_category] if l2_category_counts[l2_category] > 0 else 0
        print(f"{l2_category}: {l2_category_accuracy:.4f} ({l2_category_correct[l2_category]}/{l2_category_counts[l2_category]})")
    
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_count})")
    
    # Save results
    model_name = os.path.basename(args.model_path)
    with open(os.path.join(args.output_dir, f"{model_name}_mmstar_results.json"), 'w') as f:
        json.dump({
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_count": total_count,
            "category_accuracy": {cat: category_correct[cat] / category_counts[cat] if category_counts[cat] > 0 else 0 for cat in category_counts},
            "l2_category_accuracy": {cat: l2_category_correct[cat] / l2_category_counts[cat] if l2_category_counts[cat] > 0 else 0 for cat in l2_category_counts},
            "results": results
        }, f, indent=2)
    print(f"Results saved to {os.path.join(args.output_dir, f'{model_name}_mmstar_results.json')}")
if __name__ == "__main__":
    main()