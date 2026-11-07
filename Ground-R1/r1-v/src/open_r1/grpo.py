# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, DatasetDict
from LaVCOT.constants import DEFAULT_POINTER_PAD_TOKEN, STAGE_ONE_TEMPLATE
from accuracy_reward_functions import v7w, docVQA, Math_eureka, ocr_and_freeform, openend_datasets, deepeyes_visual_toolbox, output_json_for_selection

def compute_iou(gt_bbox, student_bbox):
    x1_gt, y1_gt = gt_bbox[0]
    x2_gt, y2_gt = gt_bbox[1]
    
    x1_st, y1_st = student_bbox[0]
    x2_st, y2_st = student_bbox[1]

    x1_inter = max(x1_gt, x1_st)
    y1_inter = max(y1_gt, y1_st)
    x2_inter = min(x2_gt, x2_st)
    y2_inter = min(y2_gt, y2_st)

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    student_area = (x2_st - x1_st) * (y2_st - y1_st)

    union_area = gt_area + student_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    iou = iou * 2

    return iou


def compute_giou(gt_bbox, student_bbox):
    x1_gt, y1_gt = gt_bbox[0]
    x2_gt, y2_gt = gt_bbox[1]
    
    x1_st, y1_st = student_bbox[0]
    x2_st, y2_st = student_bbox[1]

    x1_inter = max(x1_gt, x1_st)
    y1_inter = max(y1_gt, y1_st)
    x2_inter = min(x2_gt, x2_st)
    y2_inter = min(y2_gt, y2_st)

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    student_area = (x2_st - x1_st) * (y2_st - y1_st)

    union_area = gt_area + student_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    x1_c = min(x1_gt, x1_st)
    y1_c = min(y1_gt, y1_st)
    x2_c = max(x2_gt, x2_st)
    y2_c = max(y2_gt, y2_st)

    c_area = (x2_c - x1_c) * (y2_c - y1_c)

    giou = iou - (c_area - union_area) / c_area if c_area > 0 else iou

    giou_scaled = giou + 1
    return giou_scaled

dataset_handlers = {

    "v7w": (v7w, False),
    "gqa": (v7w, False),
    "openimages": (v7w, False),
    "CLEVR": (Math_eureka, False),
    "docvqa": (ocr_and_freeform, True),  # 需要dataset参数
    "infographicsvqa": (ocr_and_freeform, True),
    "textcap": (ocr_and_freeform, True), 
    "textvqa": (ocr_and_freeform, True), 
    "dude": (ocr_and_freeform, True),
    "sroie": (ocr_and_freeform, True), 
    "flickr30k": (ocr_and_freeform, True),
    "vsr": (openend_datasets, False),
    "cub": (openend_datasets, False),
    "visual_toolbox_v2": (deepeyes_visual_toolbox, False),
    "data_thinklite_reasoning_acc": (Math_eureka, False),
    "minio3": (deepeyes_visual_toolbox, False),
}

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    score_funcs: list[str] = field(
        # default_factory=lambda: ["bbox_stage1", "bbox_stage2", "accuracy_stage1", "accuracy_stage2", "bbox_iou_stage1", "bbox_iou_stage2", "bbox_iou_stage3"],
        default_factory=lambda: ["accuracy_reward_stage2"],
    )

    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    num_generations_stage1: Optional[int] = field(
        default=8,
        metadata={"help": "generations in stage1"},
    )

def clean_completion(content):
    content = content.split("<|im_start|>assistant\n", 1)[-1]
    if content.startswith("assistant\n"):
        content = content[len("assistant\n"):]
    return content.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()


def score_completion(content, solution, dataset_name):
    content = clean_completion(content)
    if not content or dataset_name not in dataset_handlers:
        return 0.0, content
    handler, needs_dataset_arg = dataset_handlers[dataset_name]
    solutions = solution if isinstance(solution, list) else [solution]
    best_reward, student_answer = 0.0, content
    for sol in solutions:
        sol = str(sol)
        match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
        ground_truth = match.group(1).strip() if match else sol.strip()
        if not ground_truth:
            continue
        if needs_dataset_arg:
            reward, candidate = handler(content, ground_truth, dataset_name)
        else:
            reward, candidate = handler(content, ground_truth)
        if reward >= best_reward:
            best_reward, student_answer = reward, candidate
    return best_reward, student_answer


def accuracy_reward_stage2(completions, solution, dataset, image, problem_id, **kwargs):

    completion_contents_stage2 = [completion[-1] if len(completion) >= 2 else "" for completion in completions]

    rewards = []

    for index, (content_stage2, sol, dataset_name) in enumerate(zip(completion_contents_stage2, solution, dataset)):
        content_stage2 = clean_completion(content_stage2)
        reward, student_answer = score_completion(content_stage2, sol, dataset_name)

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: tage2——{reward} -------------\n")
                f.write(f"---------------\"problem_id\": {problem_id[index]}--------------------------\n")
                f.write(f"Content_last_stage: {content_stage2}\n")
                f.write(f"Content Match Stage2: {student_answer}\n")
                f.write(f"Solution: {sol}\n")

    if os.getenv("DATA_SELECT") == "true":
        log_path = os.getenv("ACCURACY_PATH")
        output_json_for_selection(log_path, problem_id, image, dataset, rewards)

    return rewards

def accuracy_score_stage1(completions, solution, dataset, image, problem_id, **kwargs):
    return [
        score_completion(completion[0] if completion else "", sol, dataset_name)[0]
        for completion, sol, dataset_name in zip(completions, solution, dataset)
    ]


def accuracy_score_stage2(completions, solution, dataset, image, problem_id, **kwargs):
    return [
        score_completion(completion[-1] if len(completion) >= 2 else "", sol, dataset_name)[0]
        for completion, sol, dataset_name in zip(completions, solution, dataset)
    ]


def format_reward_stage1_pointerpad(completions, **kwargs):
    return [
        float(bool(completion) and clean_completion(completion[0]) == DEFAULT_POINTER_PAD_TOKEN)
        for completion in completions
    ]


def refine_times(completions, **kwargs):

    refine_times = []

    for completion in completions:
        refine_times.append(len(completion))

    return refine_times

# def bbox_reward(completions, bboxs, width, height, **kwargs):
def bbox_reward_stage2(crop_bbox_to_cal_iou_stage2, bboxs, problem_id, image, dataset, **kwargs):
    bbox_rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(crop_bbox_to_cal_iou_stage2, bboxs):
        if sol:
            try:
                content = [[int(content[0]), int(content[1]), int(content[2]), int(content[3])]]
                bbox_reward = 0.0
                student_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in content] + [(int(x2), int(y2)) for x1, y1, x2, y2 in content]
                # student_bbox = [(int(x1)*width/1000, int(y1)*height/1000) for x1, y1, x2, y2 in student_matches] + [(int(x2)*width/1000, int(y2)*height/1000) for x1, y1, x2, y2 in student_matches]
                gt_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in sol] + [(int(x2), int(y2)) for x1, y1, x2, y2 in sol]
                bbox_reward = compute_giou(gt_bbox, student_bbox)
            except Exception as e:
                # print(f"BBOXERROR:{e}---{content}")
                bbox_reward = 0.0
            if content == sol:
                bbox_reward = 2.0
        else:
            if content != []:
                bbox_reward = 2.0
            else:
                bbox_reward = 0.0
        bbox_rewards.append(bbox_reward / 2)

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} BBox reward-----------------\n")
            f.write(f"---------------\"problem_id\": {problem_id[0]}--------------------------\n")
            f.write(f"{crop_bbox_to_cal_iou_stage2}---completion_contents---{bbox_rewards}---bbox_reward\n")

    if os.getenv("DATA_SELECT_BBOX") == "true":
        log_path = os.getenv("BBOX_PATH")
        output_json_for_selection(log_path, problem_id, image, dataset, bbox_rewards)

    return bbox_rewards

def bbox_score_stage1(crop_bbox_to_cal_iou_stage1, bboxs, problem_id, image, dataset, **kwargs):
    bbox_scores = []
    for content, sol in zip(crop_bbox_to_cal_iou_stage1, bboxs):
        if sol:
            try:
                content = [[int(content[0]), int(content[1]), int(content[2]), int(content[3])]]
                bbox_score = 0.0
                student_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in content] + [(int(x2), int(y2)) for x1, y1, x2, y2 in content]
                gt_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in sol] + [(int(x2), int(y2)) for x1, y1, x2, y2 in sol]
                bbox_score = compute_giou(gt_bbox, student_bbox)
            except Exception as e:
                bbox_score = 0.0
            if content == sol:
                bbox_score = 2.0
        else:
            if content != []:
                bbox_score = 2.0
            else:
                bbox_score = 0.0
        bbox_scores.append(bbox_score / 2)

    return bbox_scores

def bbox_score_stage2(crop_bbox_to_cal_iou_stage2, bboxs, problem_id, image, dataset, **kwargs):
    bbox_scores = []
    for content, sol in zip(crop_bbox_to_cal_iou_stage2, bboxs):
        if sol:
            try:
                content = [[int(content[0]), int(content[1]), int(content[2]), int(content[3])]]
                bbox_score = 0.0
                student_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in content] + [(int(x2), int(y2)) for x1, y1, x2, y2 in content]
                gt_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in sol] + [(int(x2), int(y2)) for x1, y1, x2, y2 in sol]
                bbox_score = compute_giou(gt_bbox, student_bbox)
            except Exception as e:
                bbox_score = 0.0
            if content == sol:
                bbox_score = 2.0
        else:
            if content != []:
                bbox_score = 2.0
            else:
                bbox_score = 0.0
        bbox_scores.append(bbox_score / 2)

    return bbox_scores


def bbox_iou_stage2(crop_bbox_to_cal_iou_stage2, bboxs, problem_id, image, dataset, **kwargs):
    bbox_scores = []
    for content, sol in zip(crop_bbox_to_cal_iou_stage2, bboxs):
        if sol:
            try:
                content = [[int(content[0]), int(content[1]), int(content[2]), int(content[3])]]
                bbox_score = 0.0
                student_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in content] + [(int(x2), int(y2)) for x1, y1, x2, y2 in content]
                gt_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in sol] + [(int(x2), int(y2)) for x1, y1, x2, y2 in sol]
                bbox_score = compute_iou(gt_bbox, student_bbox)
            except Exception as e:
                bbox_score = 0.0
            if content == sol:
                bbox_score = 2.0
        else:
            if content != []:
                bbox_score = 2.0
            else:
                bbox_score = 0.0
        bbox_scores.append(bbox_score / 2)

    return bbox_scores

def bbox_iou_stage1(crop_bbox_to_cal_iou_stage1, bboxs, problem_id, image, dataset, **kwargs):
    bbox_scores = []
    for content, sol in zip(crop_bbox_to_cal_iou_stage1, bboxs):
        if sol:
            try:
                content = [[int(content[0]), int(content[1]), int(content[2]), int(content[3])]]
                bbox_score = 0.0
                student_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in content] + [(int(x2), int(y2)) for x1, y1, x2, y2 in content]
                gt_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in sol] + [(int(x2), int(y2)) for x1, y1, x2, y2 in sol]
                bbox_score = compute_iou(gt_bbox, student_bbox)
            except Exception as e:
                bbox_score = 0.0
            if content == sol:
                bbox_score = 2.0
        else:
            if content != []:
                bbox_score = 2.0
            else:
                bbox_score = 0.0
        bbox_scores.append(bbox_score / 2)

    return bbox_scores

def bbox_iou_stage3(crop_bbox_to_cal_iou_stage3, bboxs, problem_id, image, dataset, **kwargs):
    bbox_scores = []
    for content, sol in zip(crop_bbox_to_cal_iou_stage3, bboxs):
        if sol:
            try:
                content = [[int(content[0]), int(content[1]), int(content[2]), int(content[3])]]
                bbox_score = 0.0
                student_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in content] + [(int(x2), int(y2)) for x1, y1, x2, y2 in content]
                gt_bbox = [(int(x1), int(y1)) for x1, y1, x2, y2 in sol] + [(int(x2), int(y2)) for x1, y1, x2, y2 in sol]
                bbox_score = compute_iou(gt_bbox, student_bbox)
            except Exception as e:
                bbox_score = 0.0
            if content == sol:
                bbox_score = 2.0
        else:
            if content != []:
                bbox_score = 2.0
            else:
                bbox_score = 0.0
        bbox_scores.append(bbox_score / 2)

    return bbox_scores

reward_funcs_registry = {
    "accuracy": accuracy_reward_stage2,
    "format": format_reward_stage1_pointerpad,
    # "bbox": bbox_reward_stage2,
    # "bbox_stage1": bbox_score_stage1,
    # "accuracy_stage1": accuracy_score_stage1,
}

log_score_funcs_regitry = {
    # "bbox_stage2": bbox_score_stage2,
    # "accuracy_stage2": accuracy_score_stage2,
    # "bbox_score_stage1": bbox_score_stage1,
    "accuracy_reward_stage2": accuracy_score_stage2,
    "refine_times": refine_times,
    # "bbox_score_stage2": bbox_reward_stage2,
    # "bbox_iou_stage1": bbox_iou_stage1,
    # "bbox_iou_stage2": bbox_iou_stage2,
    # "bbox_iou_stage3": bbox_iou_stage3,
}



def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    score_funcs = [log_score_funcs_regitry[func] for func in script_args.score_funcs]

    if script_args.dataset_name[-6:] == '.jsonl':
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    def make_conversation_image(example):
        
        return {
            "prompt": [
                # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        # {"type": "text", "text": BBOX_QUESTION_TEMPLATE.format(Question=example["problem"])},
                        {"type": "text", "text": STAGE_ONE_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
                # {"role": "assistant", "content": [{"type": "text", "text": "(141,236),(217,287)"}]},
                # {
                #     "role": "user",
                #     "content": [
                #         {"type": "image"},
                #         # {"type": "text", "text": BBOX_QUESTION_TEMPLATE.format(Question=example["problem"], Width=example["width"], Height=example["height"])},
                #         {"type": "text", "text": "Please describe the second image."},
                #     ],
                # },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])
        
    elif "video_filename" in dataset[script_args.dataset_train_split].features:
        print("has video in dataset")
        dataset = dataset.map(make_conversation_video)

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    if last_checkpoint:
        print(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        score_funcs=score_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        num_generations_stage1=script_args.num_generations_stage1,
    )

    # Train and push the model to the Hub
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save and push to hub
    trainer.save_state(training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
