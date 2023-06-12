deepspeed --include localhost:0,1,2,3,4,5,6,7 blip2/run_pretrain_stage2.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/stage2/vitL_ada.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir pretrain/blip2_stage2_test/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path pretrain/blip2_test2/checkpoint-0195000/ckpt.pth \
    --do_lower_case --learning_rate 1e-04  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 256 \
    --ckpt_period 20000 --max_iters 200000 --warmup_steps 20000   --log_period 10  \
    --data_dir ./pretrain_datasets/ --dataset_file root_based/cc12m_cc_coco_vg_img_json.yaml --data_debug

# vicuna
deepspeed --include localhost:0,1,2,3,4,5,6,7 blip2/run_pretrain_stage2.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/stage2/vitL_ada_vicuna.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir pretrain/blip2_stage2_vicuna_8GPUS/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path pretrain/blip2_test2_vicuna_8GPUS/checkpoint-0039000/ckpt.pth \
    --do_lower_case --learning_rate 5e-05  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 256 \
    --ckpt_period 10000 --max_iters 50000 --warmup_steps 5000   --log_period 10  \
    --data_dir ./pretrain_datasets/ --dataset_file root_based/cc12m_cc_coco_vg_img_json.yaml --no_autocontrast

# for baai
deepspeed --include localhost:0,1,2,3 blip2/run_pretrain_stage2.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/stage2/vitL_ada_baai.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir pretrain/blip2_stage2_test/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path /share/project/zejunli/ckpt/BLIP2/stage1/vitL_ada_stage1_ep15.pth \
    --do_lower_case --learning_rate 1e-04  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 128 \
    --ckpt_period 20000 --max_iters 200000 --warmup_steps 20000   --log_period 10  \
    --data_dir ./pretrain_datasets/ --dataset_file cc12m_cc_coco_vg_img_baai.yaml --data_debug


# for instruct tuning on llava
deepspeed --include localhost:0,1,2,3,4,5,6,7 blip2/run_vl_instruct.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/instruct_tuning/vitL_ada_vicuna.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir pretrain/blip2_instruct_llava_vicuna_8GPUS/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path pretrain/blip2_stage2_vicuna_8GPUS/checkpoint-0050000/ckpt.pth \
    --do_lower_case --learning_rate 1e-05  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 64 \
    --ckpt_period 5000 --max_iters 10000 --warmup_steps 500   --log_period 10  \
    --data_dir ./ --dataset_file blip2/configs/instruct_dataset/llava_local.yaml --no_autocontrast

# for 4GPUs
deepspeed --include localhost:3,4,5,7 blip2/run_vl_instruct.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/instruct_tuning/vitL_ada_vicuna.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir pretrain/blip2_instruct_llava_vicuna_4GPUS/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path /root/tmp_ckpt/ckpt.pth \
    --do_lower_case --learning_rate 1e-05  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 16 \
    --ckpt_period 10000 --max_iters 20000 --warmup_steps 1000   --log_period 10  \
    --data_dir ./ --dataset_file blip2/configs/instruct_dataset/llava_local.yaml --no_autocontrast