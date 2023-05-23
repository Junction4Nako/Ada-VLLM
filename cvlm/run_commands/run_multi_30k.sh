# uni stage2 (TLM+VLM) (fr)
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/multi30k/retrieval_base_lr_xlm-r.yaml \
    --model_name_or_path pretrain/unfied_stage2_xlm-r_all_langs_8gpus/checkpoint-0240000/ckpt.pth \
    --tokenizer_name xlm-roberta-base \
    --do_train --do_lower_case --save_steps 2300 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00002 \
    --per_gpu_eval_batch_size 32  --train_language fr  --test_language fr \
    --num_train_epochs 10 --weight_decay 0.05  --image_dir_format local \
    --max_seq_length 50  --evaluate_during_training  \
    --num_captions_per_img_val 64 --num_images_per_cap_val 32  --save_metric mR \
    --output_dir output_multi30k/albef_uni_stage2_english_only/

# train on en, evaluate on other languages
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/multi30k/retrieval_base_lr_xlm-r.yaml \
    --model_name_or_path pretrain/unfied_stage2_xlm-r_all_langs_8gpus/checkpoint-0240000/ckpt.pth \
    --tokenizer_name xlm-roberta-base \
    --do_train --do_lower_case --save_steps 2300 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00002 \
    --per_gpu_eval_batch_size 32  --train_language en  --test_language all \
    --num_train_epochs 10 --weight_decay 0.05  --image_dir_format local \
    --max_seq_length 50  --evaluate_during_training  \
    --num_captions_per_img_val 64 --num_images_per_cap_val 32  --save_metric mR \
    --output_dir output_multi30k/albef_uni_stage2_english_only/

# train on all, evaluate on all languages
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/multi30k/retrieval_base_lr_xlm-r.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_init_8gpus_half/checkpoint-0120000/ckpt.pth \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ --logging_steps 40 \
    --do_train --do_lower_case --save_steps 5400 \
    --per_gpu_train_batch_size 12 --learning_rate 0.00002 \
    --per_gpu_eval_batch_size 32  --train_language all  --test_language all \
    --num_train_epochs 10 --weight_decay 0.05  --image_dir_format local \
    --max_seq_length 50  --evaluate_during_training  \
    --num_captions_per_img_val 64 --num_images_per_cap_val 32  --save_metric mR \
    --output_dir output_multi30k/albef_uni_stage2_mlm_init_half_all/  --cuda_devices 4,5,6,7

# train for xlm-r init
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/multi30k/retrieval_base_lr_xlm-r_init.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_init_abl-tlm_7gpus_half/checkpoint-0138000 \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ --logging_steps 40 \
    --do_train --do_lower_case --save_steps 3625 \
    --per_gpu_train_batch_size 12 --learning_rate 0.00002 \
    --per_gpu_eval_batch_size 32  --train_language all  --test_language all \
    --num_train_epochs 10 --weight_decay 0.05  --image_dir_format local \
    --max_seq_length 50  --evaluate_during_training  \
    --num_captions_per_img_val 32 --num_images_per_cap_val 32  --save_metric mR \
    --output_dir output_multi30k/albef_uni_stage2_mlm_init_abl-tlm_half_all/  --cuda_devices 4,5,6,7

# train for xlm-r init and freeze visual encoder
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/multi30k/retrieval_base_lr_xlm-r_init_freeze_vis.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_init_8gpus_half/checkpoint-0120000 \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ --logging_steps 40 \
    --do_train --do_lower_case --save_steps 3625 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00002 \
    --per_gpu_eval_batch_size 32  --train_language all  --test_language all \
    --num_train_epochs 10 --weight_decay 0.05  --image_dir_format local \
    --max_seq_length 50  --evaluate_during_training  \
    --num_captions_per_img_val 32 --num_images_per_cap_val 32  --save_metric mR \
    --output_dir output_multi30k/albef_uni_stage2_mlm_init_half_all_freeze_vis/  --cuda_devices 4,5,6,7

# test
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/multi30k/retrieval_base_lr_xlm-r.yaml \
    --eval_model_dir output_multi30k/albef_uni_stage2_mlm_prob25_all/checkpoint-8-65250/ \
    --tokenizer_name xlm-roberta-base  --eval_split test \
    --do_test  --do_eval --do_lower_case --image_dir_format local \
    --per_gpu_eval_batch_size 32  --test_language de --max_seq_length 50  \
    --num_captions_per_img_val 32 --num_images_per_cap_val 32  --test_language all \
    --output_dir output_multi30k/evaluation/ 

# test for xlm-r init
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/multi30k/retrieval_base_lr_xlm-r_init.yaml \
    --eval_model_dir output_multi30k/albef_uni_stage2_mlm_init_en/checkpoint-6-21000/ \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/  --eval_split test \
    --do_test  --do_eval --do_lower_case --image_dir_format local \
    --per_gpu_eval_batch_size 32  --max_seq_length 50  \
    --num_captions_per_img_val 32 --num_images_per_cap_val 32  --test_language all \
    --output_dir output_multi30k/evaluation/  --cuda_devices 0,1,2,3,4,5

# output_multi30k/albef_uni_stage2_mlm_init_fr/checkpoint-9-9000/ \