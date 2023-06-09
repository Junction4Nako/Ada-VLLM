deepspeed --include localhost:0,1,2,3,4,5,6,7 blip2/run_pretrain_stage1.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/stage1/vitL_ada.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir /root/blip2_test/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path /remote-home/zjli/LLM/checkpoints/blip2/blip2_pretrained_vitL_ada.pth \
    --do_lower_case --learning_rate 1e-04  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 512 \
    --ckpt_period 6000 --max_iters 90000 --warmup_steps 9000   --log_period 10  \
    --data_dir ./pretrain_datasets/ --dataset_file root_based/cc12m_cc_coco_vg_img.yaml

# bug test
deepspeed --include localhost:4,5,6,7 blip2/run_pretrain_stage1.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/stage1/vitL_ada.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir /root/blip2_test2/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path /remote-home/zjli/LLM/checkpoints/blip2/blip2_pretrained_vitL_ada.pth \
    --do_lower_case --learning_rate 1e-04  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 256 \
    --ckpt_period 100 --max_iters 90000 --warmup_steps 9000   --log_period 10  \
    --data_dir ./pretrain_datasets/ --dataset_file root_based/cc12m_cc_coco_vg_img.yaml --data_debug

# baai test
deepspeed --include localhost:0,1,2,3 blip2/run_pretrain_stage1.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/stage1/vitL_ada.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir pretrain/blip2_test2/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path /share/project/zejunli/ckpt/BLIP2/blip2_pretrained_vitL_ada.pth \
    --do_lower_case --learning_rate 1e-04  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 256 \
    --ckpt_period 100 --max_iters 90000 --warmup_steps 9000   --log_period 10  \
    --data_dir ./pretrain_datasets/ --dataset_file cc12m_cc_coco_vg_img_baai.yaml --data_debug

# 226 root loading
deepspeed --include localhost:0,1,2,3,4,5,6,7 blip2/run_pretrain_stage1.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/stage1/vitL_ada.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir pretrain/blip2_test2/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path /root/ckpt/blip2/blip2_pretrained_vitL_ada.pth \
    --do_lower_case --learning_rate 1e-04  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 1024 \
    --ckpt_period 13000 --max_iters 200000 --warmup_steps 20000   --log_period 10  \
    --data_dir ./pretrain_datasets/ --dataset_file root_based/cc12m_cc_coco_vg_img_json.yaml --data_debug

# 226 for vicuna
deepspeed --include localhost:0,1,2,3,4,5,6,7 blip2/run_pretrain_stage1.py \
    --deepspeed_config oscar/tmp_config.json --model_config blip2/configs/stage1/vitL_ada.yaml \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 --output_dir pretrain/blip2_test2_vicuna_8GPUS/  \
    --tokenizer_name huggyllama/llama-7b --model_name_or_path /root/ckpt/blip2/blip2_pretrained_vitL_ada_vicuna.pth \
    --do_lower_case --learning_rate 1e-04  --do_train --deepspeed \
    --max_seq_length 35  --max_seq_length_txt 50  --on_memory  --num_workers 4 --drop_out 0.1  --train_batch_size 1024 \
    --ckpt_period 13000 --max_iters 39000 --warmup_steps 2000   --log_period 10  \
    --data_dir ./pretrain_datasets/ --dataset_file root_based/cc12m_cc_coco_vg_img_json.yaml