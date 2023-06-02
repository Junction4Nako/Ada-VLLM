# !/bin/bash
cd /share/project/zejunli/code/AdaBLIP/Ada-VLLM

conda activate /share/project/zejunli/envs

deepspeed --include localhost:0,1,2,3,4,5,6,7 blip2/run_pretrain_stage1.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/stage1/vitL_ada_baai.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir pretrain/blip2_test2_vicuna_8GPUS/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path /share/project/zejunli/ckpt/BLIP2/blip2_pretrained_vitL_ada_vicuna.pth \
    --do_lower_case --learning_rate 1e-04  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 1024 \
    --ckpt_period 13000 --max_iters 200000 --warmup_steps 20000   --log_period 10  \
    --data_dir ./pretrain_datasets/ --dataset_file cc12m_cc_coco_vg_img_baai.yaml