python3 cvlm/run_xFlickrCO_albef.py \
    --albef_config albef/configs/xFlickrCO/retrieval_base_xFlickrCO_xlm-r.yaml \
    --model_name_or_path pretrain/stage1_xlm-r_all_langs/checkpoint-0120000/ckpt.pth \
    --tokenizer_name xlm-roberta-base \
    --do_train --do_lower_case --save_steps 2200 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00004 \
    --per_gpu_eval_batch_size 32  --test_lang all \
    --num_train_epochs 10 --weight_decay 0.05 \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 16 --num_images_per_cap_val 16 \
    --output_dir output_xFlickrCO/albef_stage1_xlmr/  --save_metric r1  --cuda_devices 0,1,2,3,4,5,7

# stage 2
python3 cvlm/run_xFlickrCO_albef.py \
    --albef_config albef/configs/xFlickrCO/retrieval_base_xFlickrCO_xlm-r.yaml \
    --model_name_or_path pretrain/stage2_xlmr_trans_all_lang_vis_freeze/checkpoint-0135000 \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ \
    --do_train --do_lower_case --save_steps 4500 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00004 \
    --per_gpu_eval_batch_size 32  --test_lang all \
    --num_train_epochs 10 --weight_decay 0.05 \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 16 --num_images_per_cap_val 16 \
    --output_dir output_xFlickrCO/albef_stage2_xlmr/  --save_metric r1  --cuda_devices 0,1,3

# unified stage 2 (TLM + VLM)
python3 cvlm/run_xFlickrCO_albef.py \
    --albef_config albef/configs/xFlickrCO/retrieval_base_xFlickrCO_xlm-r.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_cond_prob25/checkpoint-0240000 \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ \
    --do_train --do_lower_case --save_steps 2000 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00004 \
    --per_gpu_eval_batch_size 32  --test_lang all \
    --num_train_epochs 10 --weight_decay 0.05  --image_dir_format local \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 16 --num_images_per_cap_val 16 \
    --output_dir output_xFlickrCO/albef_uni_stage2_xlmr_vlm_tlm_mlm_prob25/  --save_metric r1

# freeze single
python3 cvlm/run_xFlickrCO_albef.py \
    --albef_config albef/configs/xFlickrCO/retrieval_base_xFlickrCO_xlm-r_freeze_single.yaml \
    --model_name_or_path pretrain/stage1_xlm-r_all_langs/checkpoint-0120000/ckpt.pth \
    --tokenizer_name xlm-roberta-base \
    --do_train --do_lower_case --save_steps 2200 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00004 \
    --per_gpu_eval_batch_size 32  --test_lang all \
    --num_train_epochs 10 --weight_decay 0.05 \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 16 --num_images_per_cap_val 16 \
    --output_dir output_xFlickrCO/albef_stage1_xlmr_freeze_single/  --save_metric r1

# for xlm-r init
python3 cvlm/run_retrieval_albef_wit.py \
    --albef_config albef/configs/wit/retrieval_base_wit_xlm-r_init.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_init_8gpus/checkpoint-0240000/ \
    --tokenizer_name xlm-roberta-base --image_dir_format local \
    --do_train --do_lower_case --save_steps 10000 \
    --per_gpu_train_batch_size 12 --learning_rate 0.00004 \
    --per_gpu_eval_batch_size 32  --test_lang all \
    --num_train_epochs 10 --weight_decay 0.05 \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 16 --num_images_per_cap_val 16 \
    --output_dir output_wit/albef_uni_stage2_xlmr_init/  --save_metric r1 --cuda_devices 2,3

# do test
python3 cvlm/run_xFlickrCO_albef.py \
    --albef_config albef/configs/xFlickrCO/retrieval_base_xFlickrCO_xlm-r.yaml \
    --eval_model_dir output_xFlickrCO/albef_uni_stage2_xlmr_vlm_tlm_mlm_prob25/checkpoint-4-10000/ \
    --tokenizer_name xlm-roberta-base  --eval_split test \
    --do_test  --do_eval --do_lower_case --image_dir_format local \
    --per_gpu_eval_batch_size 64  --test_lang all --max_seq_length 50  \
    --num_captions_per_img_val 16 --num_images_per_cap_val 16 \
    --output_dir output_xFlickrCO/evaluation/  --cuda_devices 2,3

# do test for xlm-r init
python3 cvlm/run_retrieval_albef_wit.py \
    --albef_config albef/configs/wit/retrieval_base_wit_xlm-r_init.yaml \
    --eval_model_dir pretrain/unified_stage2_xlm-r_all_langs_mono_init_8gpus/checkpoint-0240000/ \
    --tokenizer_name xlm-roberta-base  --eval_split translate_test \
    --do_test  --do_eval --do_lower_case --image_dir_format local \
    --per_gpu_eval_batch_size 16  --test_lang all --max_seq_length 50  \
    --num_captions_per_img_val 16 --num_images_per_cap_val 16 \
    --output_dir output_wit/evaluation/  --cuda_devices 2,3

# using extra txt parallel dataset
python3 cvlm/run_xFlickrCO_albef.py \
    --albef_config albef/configs/xFlickrCO/retrieval_base_xFlickrCO_xlm-r_freeze_vis.yaml \
    --model_name_or_path pretrain/stage1_xlm-r_all_langs/checkpoint-0120000/ckpt.pth \
    --tokenizer_name xlm-roberta-base  --add_txt  --txt_dataformat transformers \
    --data_dir ./pretrain_datasets/ --txt_dataset_file wikimatrix_simplified.yaml \
    --train_batch_size_txt 1024  --max_seq_length_txt 50 \
    --do_train --do_lower_case --save_steps 2200 \
    --per_gpu_train_batch_size 32 --learning_rate 0.00004 \
    --per_gpu_eval_batch_size 64  --test_lang all \
    --num_train_epochs 10 --weight_decay 0.05 \
    --max_seq_length 50  --evaluate_during_training \
    --num_captions_per_img_val 16 --num_images_per_cap_val 16 \
    --output_dir output_xFlickrCO/albef_stage1_xlmr_freeze_vis_parallel_txt/  --save_metric r1