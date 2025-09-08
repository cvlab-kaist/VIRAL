import os
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import re

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/viral-7b")
    parser.add_argument("--model-base", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--jsonl-path", type=str, default="./playground/benchmarks/POPE/pope_popular.jsonl")
    parser.add_argument("--image-folder", type=str, default="./playground/benchmarks/COCO2014/val2014")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    return parser.parse_args()

def extract_yes_no(text):
    """응답에서 yes 또는 no를 추출합니다."""
    text = text.lower()
    
    # 간단한 패턴 매칭으로 yes/no 추출
    if re.search(r'\byes\b', text):
        return "yes"
    elif re.search(r'\bno\b', text):
        return "no"
    
    # 더 복잡한 패턴 처리
    affirmative = ["yes", "yeah", "correct", "right", "true", "indeed", "affirmative"]
    negative = ["no", "nope", "not", "negative", "incorrect", "false"]
    
    # 첫 문장에 집중
    first_sentence = text.split('.')[0]
    
    for word in affirmative:
        if word in first_sentence:
            return "yes"
    
    for word in negative:
        if word in first_sentence:
            return "no"
    
    # 기본값 반환
    print(f"Warning: Could not extract yes/no from: '{text}'")
    return "unknown"

def main():
    args = parse_args()
    
    # 모델 로드
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
    # JSONL 파일 로드
    with open(args.jsonl_path, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    print(f"Loaded {len(examples)} examples from {args.jsonl_path}")
    
    # 평가 준비
    total = len(examples)
    correct = 0
    results = []
    
    # 평가 루프
    for example in tqdm(examples):
        # 이미지 로드
        image_path = os.path.join(args.image_folder, example["image"])
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # 대화 설정
        conv = conv_templates[conv_mode].copy()
        question = example["text"].strip()
        # question += " Base your answer on reasoning, but answer with only 'yes' or 'no'."
        question += " Answer with only 'yes' or 'no'."
        # breakpoint()
        
        # 프롬프트 구성
        inp = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 토큰화
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # 생성 설정
        temperature = args.temperature
        max_new_tokens = args.max_new_tokens
        
        # 추론 실행
        with torch.inference_mode():
            outputs = model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        
        # 출력 처리
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 응답 추출
        if conv.sep_style == SeparatorStyle.TWO:
            response = output_text.split(conv.sep2)[-1].strip()
        elif conv.sep_style == SeparatorStyle.LLAMA_2:
            response = output_text.split(conv.roles[1] + ":")[-1].strip()
        else:
            response = output_text.split(conv.roles[1] + ":")[-1].strip()
        
        # yes/no 추출
        pred = extract_yes_no(response)
        
        # 결과 저장
        is_correct = (pred == example["label"])
        if is_correct:
            correct += 1
        
        results.append({
            "question_id": example["question_id"],
            "image": example["image"],
            "question": question,
            "prediction": pred,
            "label": example["label"],
            "correct": is_correct,
            "full_response": response
        })
    
    # 정확도 계산
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    
    # 결과 저장
    
    version = "adversarial" if "adversarial" in args.jsonl_path else "popular" if "popular" in args.jsonl_path else "random"
    result_name = f"{os.path.basename(args.model_path)}_{version}_results.json"
    with open(result_name, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)

if __name__ == "__main__":
    main()