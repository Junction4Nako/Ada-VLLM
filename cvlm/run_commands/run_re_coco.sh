# human written of coco-cn
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/retrieval_base_coco_zh_lr_xlm-r.yaml \
    --model_name_or_path pretrain/stage1_xlm-r_sub_zh/checkpoint-0120000/ckpt.pth \
    --tokenizer_name xlm-roberta-base \
    --do_train --do_lower_case --save_steps 200 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00004 \
    --per_gpu_eval_batch_size 32  \
    --num_train_epochs 10 --weight_decay 0.05 \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 128 --num_images_per_cap_val 64 \
    --output_dir output_coco_cn/albef_stage1_xlmr_sub_zh/  --save_metric mR  --cuda_devices 0,1,2,3,4,5,7

# test if new script is correct
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/retrieval_base_fk30_zh.yaml \
    --model_name_or_path /remote-home/zjli/CVLM/ckpt/stage1_tlm/checkpoint-0090000/ckpt.pth \
    --tokenizer_name xlm-mlm-tlm-xnli15-1024 \
    --do_train --do_lower_case --save_steps 2400 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00004 \
    --per_gpu_eval_batch_size 32  \
    --num_train_epochs 10 --weight_decay 0.05 \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 128 --num_images_per_cap_val 64 \
    --output_dir output_fk_cn/albef_stage1_tlm/

# test retrieval on coco-cn
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/retrieval_base_coco_zh.yaml \
    --eval_model_dir output_coco_cn/albef_stage2_test/checkpoint-8-2800/ \
    --tokenizer_name xlm-mlm-tlm-xnli15-1024 \
    --do_test --do_eval --eval_split test \
    --per_gpu_eval_batch_size 64 --max_seq_length 50 \
    --num_captions_per_img_val 128 --num_images_per_cap_val 128 \
    --output_dir output_coco_cn/evaluation/

# xlm-r based train
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/retrieval_base_coco_zh_lr_xlm-r.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_cond_prob25/checkpoint-0240000/ \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ \
    --do_train --do_lower_case --save_steps 310 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00004 \
    --per_gpu_eval_batch_size 32  \
    --num_train_epochs 10 --weight_decay 0.05 \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 128 --num_images_per_cap_val 64 \
    --output_dir output_coco_cn/albef_stage2_prob25/ --cuda_devices 0,2

# xlm-r based test
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/retrieval_base_coco_zh_lr_xlm-r.yaml \
    --eval_model_dir output_coco_cn/albef_stage2_128mlm/checkpoint-7-2480/ \
    --tokenizer_name xlm-roberta-base \
    --do_test --do_eval --eval_split test \
    --per_gpu_eval_batch_size 64 --max_seq_length 50 \
    --num_captions_per_img_val 32 --num_images_per_cap_val 32 \
    --output_dir output_coco_cn/evaluation/


# xlm-r_init based train
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/mscoco/retrieval_base_coco_zh_lr_xlm-r_init.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_init_8gpus/checkpoint-0240000/ \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ \
    --do_train --do_lower_case --save_steps 150 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00004 \
    --per_gpu_eval_batch_size 32  \
    --num_train_epochs 10 --weight_decay 0.05 \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 64 --num_images_per_cap_val 32 \
    --output_dir output_coco_cn/albef_uni_stage2_init/ --cuda_devices 0,2

# xlm-r_init based test
python3 cvlm/run_retrieval_albef_un.py \
    --albef_config albef/configs/mscoco/retrieval_base_coco_zh_lr_xlm-r_init.yaml \
    --eval_model_dir output_coco_cn/albef_uni_stage2_init/checkpoint-9-1500/ \
    --tokenizer_name xlm-roberta-base \
    --do_test --do_eval --eval_split test \
    --per_gpu_eval_batch_size 64 --max_seq_length 50 \
    --num_captions_per_img_val 16 --num_images_per_cap_val 16 \
    --output_dir output_coco_cn/evaluation/