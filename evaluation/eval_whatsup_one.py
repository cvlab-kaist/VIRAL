import json
import os
import argparse
from PIL import Image
import torch
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
import re

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description="Run LLaVA model on QA image dataset and save results.")
parser.add_argument("--model_path", required=True, help="Path to the pretrained LLaVA model.")
parser.add_argument(
    "--qa_json_path",
    default="./playground/benchmarks/AdaptVis/data/coco_qa_one_obj.json",
    help="Path to the QA JSON file."
)
parser.add_argument(
    "--adaptvis_json_path",
    default="./playground/benchmarks/AdaptVis/prompts/COCO_QA_one_obj_with_answer_four_options.jsonl",
    help="Path to the QA JSON file."
)
parser.add_argument(
    "--image_folder",
    default="./playground/benchmarks/AdaptVis/data/val2017",
    help="Path to the folder containing COCO images."
)
parser.add_argument(
    "--output_dir",
    default="./results/whatsup_results",
    help="Output directory to save results."
)
args = parser.parse_args()

# === AUTO-CONSTRUCT OUTPUT PATH ===
model_name = os.path.basename(args.model_path.rstrip("/"))
words = model_name
# suffix = words[-1]
# prefix = next((w for w in words if "repa" in w or "llava" in w), "llava")
output_json_name = f"llava_qa_results_{model_name}.json"
output_json_path = os.path.join(args.output_dir, output_json_name)

# === LOAD MODEL ===
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

model.vra_loss = False
model.residual = False
model.diffusion_loss = False
model.residual_source = "vision_tower"
model.residual_target_layers = [16]

# === LOAD QA DATA ===
with open(args.qa_json_path, 'r') as f:
    qa_data = json.load(f)

with open(args.adaptvis_json_path, 'r') as f:
    adaptvis_data = [json.loads(line) for line in f]

assert len(qa_data) == len(adaptvis_data), "Mismatch in number of questions and adaptvis data."

results = []
right_count = 0
wrong_count = 0
else_count = 0

for idx, (image_id, question1, question2) in enumerate(tqdm(qa_data)):
    image_file = os.path.join(args.image_folder, f"{image_id:012d}.jpg")
    if not os.path.exists(image_file):
        print(f"Image not found: {image_file}")
        continue

    raw_image = Image.open(image_file).convert("RGB")
    image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].half().cuda()

    conv = conv_templates[conv_mode].copy()
    adaptvis_question = adaptvis_data[idx]['question']
    match = re.search(r"USER:\s*(.*?\?)", adaptvis_data[idx]['question'])
    # prompt = f"Which of the following statement is correct? 1) {question1} 2) {question2}."
    prompt = match.group(1) + 'Answer with a single word from left, right, top, or bottom.\n'

    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            top_p=0.7,
            max_new_tokens=100,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        print(f"Image ID: {image_id}, Answer: {answer}")
        check = answer.split('ASSISTANT')[-1]
        adaptvis_answer = adaptvis_data[idx]['answer'][0]
        print("Prompt:", prompt)
        print("check:", check)
        print("AdaptVis Answer:", adaptvis_answer)
        if (adaptvis_answer.lower() in check.lower()) or (check.lower() in adaptvis_answer.lower()):
            right_count += 1
        else:
            wrong_count += 1
        print(f"Right: {right_count}, Wrong: {wrong_count}, Else: {else_count}")

    results.append({
        "image_id": image_id,
        "question": prompt,
        "model_answer": answer,
        "adaptvis_answer": adaptvis_answer,
    })

# === SAVE RESULTS ===
with open(output_json_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open(output_json_path.replace('.json', '.txt'), 'w') as f:
    f.write(f"Right: {right_count}, Wrong: {wrong_count}, Else: {else_count}\n")

print("Done! Saved to", output_json_path)