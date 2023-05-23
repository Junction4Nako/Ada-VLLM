# raw ALBEF on GQA
python cvlm/run_xVNLI_albef.py \
    --albef_config albef/configs/xVNLI/cls_base_xVNLI_xlm-r_freeze_single.yaml \
    --model_name_or_path pretrain/stage1_xlm-r_all_langs/checkpoint-0120000/ \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ \
    --do_train --do_lower_case --save_epoch 1 \
    --per_gpu_train_batch_size 128 --learning_rate 0.00002 \
    --per_gpu_eval_batch_size 256  --test_lang all \
    --num_train_epochs 10 --weight_decay 0.05  --gradient_accumulation_steps 1 \
    --max_seq_length 40  --evaluate_during_training  --logging_steps 20  --time_debug \
    --output_dir output_xVNLI/stage1_xlmr_freeze_single/  --image_dir_format remote --cuda_devices 0,1,2,3

# distributed version
python3 -m torch.distributed.launch --nproc_per_node=4 cvlm/run_xGQA_albef.py \
    --albef_config albef/configs/qa_base_xGQA.yaml \
    --model_name_or_path /remote-home/zjli/CVLM/ckpt/pretrained/ALBEF.pth \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/txt_models/bert-base-uncased \
    --do_train --do_lower_case --save_epoch 1 \
    --per_gpu_train_batch_size 32 --learning_rate 0.00003 \
    --per_gpu_eval_batch_size 128  --ans2label_map /remote-home/zjli/CVLM/datasets/GQA/label_map/trainval_all_ans2label.pkl \
    --num_train_epochs 5 --weight_decay 0.05 \
    --max_seq_length 40  --evaluate_during_training  --logging_steps 2 \
    --output_dir output_xGQA/raw_albef_en/  --cuda_devices 0,1,2,3

# stage2 for 5 languages
python cvlm/run_xVNLI_albef.py \
    --albef_config albef/configs/xVNLI/cls_base_xVNLI_xlm-r_freeze_single.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_128txt/checkpoint-0240000/ \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ \
    --do_train --do_lower_case --save_epoch 1 \
    --per_gpu_train_batch_size 64 --learning_rate 0.00002 \
    --per_gpu_eval_batch_size 128  --test_lang all \
    --num_train_epochs 10 --weight_decay 0.05  --gradient_accumulation_steps 1 \
    --max_seq_length 40  --evaluate_during_training  --logging_steps 20  --time_debug \
    --output_dir output_xVNLI/uni_stage2_xlmr_mono/  --image_dir_format local --cuda_devices 0,1,2,3,4,5,7

# for initialized from xlm-r
python cvlm/run_xVNLI_albef.py \
    --albef_config albef/configs/xVNLI/cls_base_xVNLI_xlm-r_init_freeze_single.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_128txt/checkpoint-0240000/ \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ \
    --do_train --do_lower_case --save_epoch 1 \
    --per_gpu_train_batch_size 64 --learning_rate 0.00002 \
    --per_gpu_eval_batch_size 128  --test_lang all \
    --num_train_epochs 10 --weight_decay 0.05  --gradient_accumulation_steps 1 \
    --max_seq_length 40  --evaluate_during_training  --logging_steps 20  --time_debug \
    --output_dir output_xVNLI/uni_stage2_xlmr_mono/  --image_dir_format local

# for UC2
python cvlm/run_xVNLI_uniter.py \
    --uniter_config albef/configs/xVNLI/cls_base_xVNLI_UC2.yaml \
    --model_name_or_path /remote-home/zjli/tmp_data/UC2_DATA/pretrain/uc2/ckpt/model_step_200000.pt \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r/ \
    --do_train --do_lower_case --save_epoch 1 \
    --per_gpu_train_batch_size 64 --learning_rate 0.00002 \
    --per_gpu_eval_batch_size 128  --test_lang all \
    --num_train_epochs 10 --weight_decay 0.05  --gradient_accumulation_steps 1 \
    --max_seq_length 40  --max_img_seq_length 35 --evaluate_during_training  --logging_steps 20  --time_debug \
    --output_dir output_xVNLI/uc2/  --image_dir_format local --cuda_devices 0,1,3

# do test
python3 cvlm/run_xVNLI_albef.py \
    --albef_config albef/configs/xVNLI/cls_base_xVNLI_xlm-r_freeze_single.yaml \
    --eval_model_dir output_xNLI/uni_stage2_xlmr_mono_snli/checkpoint-4-5365/ \
    --tokenizer_name xlm-roberta-base --image_dir_format local \
    --do_test --do_eval --eval_split test \
    --per_gpu_eval_batch_size 32 --max_seq_length 50 \
    --test_lang all --output_dir output_xVNLI/evaluation/  --cuda_devices 0,1,2,3,4,5,7


# test for UC2
python3 cvlm/run_xVNLI_uniter.py \
    --uniter_config albef/configs/xVNLI/cls_base_xVNLI_UC2.yaml \
    --eval_model_dir output_xVNLI/uc2/checkpoint-4-13750/ \
    --tokenizer_name xlm-roberta-base --image_dir_format local \
    --do_test --do_eval --eval_split test \
    --per_gpu_eval_batch_size 32 --max_seq_length 50 \
    --test_lang all --output_dir output_xVNLI/evaluation/  --cuda_devices 0,1,3

# translate test
python3 cvlm/run_xVNLI_albef.py \
    --albef_config albef/configs/xVNLI/cls_base_xVNLI_xlm-r_freeze_single.yaml \
    --eval_model_dir output_xVNLI/stage1_xlmr_freeze_single/checkpoint-3-4228/ \
    --tokenizer_name xlm-roberta-base --image_dir_format remote \
    --do_test --do_eval --eval_split translate_test \
    --per_gpu_eval_batch_size 64 --max_seq_length 50 \
    --test_lang all --output_dir output_xVNLI/evaluation/