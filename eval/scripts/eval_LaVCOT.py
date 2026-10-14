import argparse
import json
import os
import re
from collections import defaultdict

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor

from dist_utils import get_rank, get_world_size, init_distributed_mode, CustomDataset_JSONL
from LaVCOT.inference import inference
from LaVCOT.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_dataset", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--max_pixels", type=int, default=2000000)
    return parser.parse_args()


def resize_image(image, max_pixels):
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = (max_pixels / (width * height)) ** 0.5
    return image.resize((int(width * scale), int(height * scale)))


def extract_answer(text):
    return re.split(r"<\|im_end\|>", text, maxsplit=1)[0].strip()


def load_model_and_processor(args, rank):
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": rank},
        attn_implementation="flash_attention_2",
    ).eval()
    return model, processor, processor.tokenizer


def make_conversation(image, instruction):
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ]},
    ]


def score_answer(answer, solution):
    answer = answer.strip().lower()
    solution = str(solution).strip().lower()
    return int(bool(answer and solution) and answer[0] == solution[0])


def evaluate(args, rank, world_size):
    model, processor, tokenizer = load_model_and_processor(args, rank)
    dataset = CustomDataset_JSONL(args.eval_dataset)
    dataset.set_rank_and_world_size(rank, world_size)
    results = []

    for item in dataset:
        image_path = os.path.join(args.dataset_path, item["Image"])
        image = resize_image(Image.open(image_path).convert("RGB"), args.max_pixels)
        problem = item["problem"]
        locate_prompt = (
            problem
            + " Your task is to locate the region that corresponds to the question. "
            + "To indicate the region, output the relevant visual pointer token."
        )
        answer_prompt = problem + " Refer to the region from the previous turn and provide the final answer."

        first = inference(
            make_conversation(image, locate_prompt),
            model,
            tokenizer,
            processor,
            topk=3,
            conv_iteration=1,
        )

        bbox_list = first.get("bboxes") or []
        if bbox_list:
            x0, y0, x1, y1 = bbox_list[0]
            width, height = image.size
            crop_box = (
                max(0, int(x0 * width)),
                max(0, int(y0 * height)),
                min(width, int(x1 * width)),
                min(height, int(y1 * height)),
            )
            crop = image.crop(crop_box) if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1] else image
        else:
            crop = image

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": locate_prompt},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "<|pointer_pad|>"}]},
            {"role": "user", "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": answer_prompt},
            ]},
        ]
        second = inference(
            conversation,
            model,
            tokenizer,
            processor,
            topk=0,
            conv_iteration=2,
        )
        answer = extract_answer(second["output_text"] or "")
        item["generation_list"] = [first["output_text"], second["output_text"]]
        item["score"] = score_answer(answer, item.get("solution", ""))
        results.append(item)
    return results


def main():
    args = parse_args()
    args.distributed = True
    args.dist_url = "env://"
    init_distributed_mode(args)
    rank, world_size = get_rank(), get_world_size()
    local_results = evaluate(args, rank, world_size)

    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results)
        all_results = [item for rank_results in gathered for item in rank_results]
    else:
        all_results = local_results

    if rank == 0:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as output:
            for item in all_results:
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts = defaultdict(lambda: [0, 0])
        for item in all_results:
            category = item.get("Category", "unknown")
            counts[category][0] += item.get("score", 0)
            counts[category][1] += 1
        for category, (correct, total) in counts.items():
            print(f"{category}: accuracy={correct / total:.4f} ({correct}/{total})")
        correct = sum(value[0] for value in counts.values())
        total = sum(value[1] for value in counts.values())
        print(f"Overall: accuracy={correct / total:.4f}" if total else "Overall: no examples")
        print(args.output_file)


if __name__ == "__main__":
    main()
