
cd r1-v

# python train_data/prepare_data.py --prefix /vast/users/xiaodan/haozezhao/data/Ground-R1/Deepeyes_Dataset/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MODE="false"
export DATA_SELECT="false"
export DATA_SELECT_BBOX="false" 
export WANDB_DISABLED=false

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    src/open_r1/grpo.py \
    --output_dir "/vast/users/xiaodan/haozezhao/model/checkpoints/qwen25vl_ground_r1" \
    --model_name_or_path "/vast/users/xiaodan/haozezhao/model/checkpoints/qwen25vl_sft" \
    --dataset_name "train_data/train_deepeyes.jsonl" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 8192 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name stage2_rethink \
    --save_steps 500 \
    --save_total_limit 10 \
    --max_grad_norm 30 \
    --save_only_model false \
    --beta 0.4 \
    --num_generations_stage1 8 \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
# --debug underflow_overflow \
# --max_pixels 401408 \
# --num_generations_stage1 default 8
