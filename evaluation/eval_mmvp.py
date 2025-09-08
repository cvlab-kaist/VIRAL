import os
import csv
import re
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/viral-7b")
    parser.add_argument("--model-base", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--csv-path", type=str, default="./playground/benchmarks/MMVP/Questions.csv")
    parser.add_argument("--image-folder", type=str, default="./playground/benchmarks/MMVP/Images")
    parser.add_argument("--image-format", type=str, default="{index}.jpg", help="Format of image filenames with {index} placeholder")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    return parser.parse_args()

def extract_option_choice(text, options_str):
    """응답에서 선택지(A/B)를 추출합니다."""
    text = text.lower()
    
    # 직접적인 옵션 문자 확인 - 새 프롬프트 형식에 맞춤
    if re.search(r'\ba\b', text) and not re.search(r'\bb\b', text):
        return "(a)"
    elif re.search(r'\bb\b', text) and not re.search(r'\ba\b', text):
        return "(b)"
    
    # 첫 문장에서 옵션 문자 확인
    first_sentence = text.split('.')[0].lower()
    if 'a' in first_sentence and 'b' not in first_sentence:
        return "(a)"
    if 'b' in first_sentence and 'a' not in first_sentence:
        return "(b)"
    
    # 선택지 파싱 - 원래 형식과 새 형식 모두 처리
    option_a_content = ""
    option_b_content = ""
    
    # 원래 형식 (a) ... (b) ...
    options_match = re.search(r'\(a\)\s*([^()]+)\s*\(b\)\s*([^()]+)', options_str.lower())
    if options_match:
        option_a_content = options_match.group(1).strip()
        option_b_content = options_match.group(2).strip()
    
    # 새 형식 A. ... B. ...
    else:
        options_match = re.search(r'A\.\s*([^AB]+)\s*B\.\s*([^AB]+)', options_str)
        if options_match:
            option_a_content = options_match.group(1).strip()
            option_b_content = options_match.group(2).strip()
    
    # 선택지 내용 기반 매칭
    if option_a_content and option_b_content:
        if option_a_content in text:
            # option_a가 포함되어 있고 option_b가 포함되어 있지 않거나,
            # option_a가 더 빨리 언급됨
            if option_b_content not in text or text.find(option_a_content) < text.find(option_b_content):
                return "(a)"
        
        if option_b_content in text:
            # option_b가 포함되어 있고 option_a가 포함되어 있지 않거나,
            # option_b가 더 빨리 언급됨
            if option_a_content not in text or text.find(option_b_content) < text.find(option_a_content):
                return "(b)"
    
    # 키워드 기반 추출 - 확장
    if any(word in text for word in ["first", "former", "a)", "a)", "a.", "option a", "choice a"]):
        return "(a)"
    if any(word in text for word in ["second", "latter", "b)", "b)", "b.", "option b", "choice b"]):
        return "(b)"
    
    # 빈도수 기반 확인 (마지막 수단)
    a_count = len(re.findall(r'\ba\b', text))
    b_count = len(re.findall(r'\bb\b', text))
    
    if a_count > b_count:
        return "(a)"
    elif b_count > a_count:
        return "(b)"
    
    # 기본값
    print(f"Warning: Could not extract choice from: '{text}' for options: {options_str}")
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
    model.residual = False
    model.residual_target_layers = [16]
    
    # CSV 파일 로드
    examples = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append({
                "index": row["Index"],
                "question": row["Question"],
                "options": row["Options"],
                "correct_answer": row["Correct Answer"]
            })
    
    print(f"Loaded {len(examples)} examples from {args.csv_path}")
    
    # 평가 준비
    results = []
    
    # 평가 루프
    for example in tqdm(examples):
        # 이미지 로드
        image_filename = args.image_format.format(index=example["index"])
        image_path = os.path.join(args.image_folder, image_filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # 대화 설정
        conv = conv_templates[conv_mode].copy()
        question = example["question"].strip()
        options = example["options"].strip()
        options = options.replace("(a)", "A.").replace("(b)", "\nB.").replace(" \n", "\n")
        
        question = f"{question}\n{options}\nBase your answer on reasoning, but answer with the option's letter from the given choices directly."
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
        
        # 선택지 추출
        pred = extract_option_choice(response, example["options"])
        
        # 결과 저장
        is_correct = (pred.strip() == example["correct_answer"].strip())
        
        results.append({
            "index": example["index"],
            "question": example["question"],
            "options": example["options"],
            "prediction": pred,
            "correct_answer": example["correct_answer"],
            "correct": is_correct,
            "full_response": response
        })
    
    # 쌍으로 정확도 계산
    question_pairs = defaultdict(list)
    
    # 질문별로 결과 그룹화
    for result in results:
        question_pairs[result["question"]].append(result)
    
    # 쌍 검증
    total_pairs = 0
    correct_pairs = 0
    pair_results = []
    
    for question, pair in question_pairs.items():
        # 쌍이 완전하지 않으면 건너뜀
        if len(pair) != 2:
            print(f"Warning: Incomplete pair for question '{question}'. Found {len(pair)} results.")
            continue
        
        total_pairs += 1
        pair_correct = all(item["correct"] for item in pair)
        
        if pair_correct:
            correct_pairs += 1
        
        pair_results.append({
            "question": question,
            "items": pair,
            "correct": pair_correct
        })
    
    # 쌍 기준 정확도 계산
    pair_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
    print(f"\nPair Accuracy: {pair_accuracy:.4f} ({correct_pairs}/{total_pairs})")
    
    # JSON으로 결과 저장
    model_name = os.path.basename(args.model_path)
    with open(f"mmvp_eval_results_{model_name}.json", 'w') as f:
        json.dump({
            "pair_accuracy": pair_accuracy,
            "individual_accuracy": sum(r["correct"] for r in results) / len(results) if results else 0,
            "pair_results": pair_results,
            "individual_results": results
        }, f, indent=2)
    
    print(f"Results saved to mmvp_eval_results_{model_name}.json")

if __name__ == "__main__":
    main()