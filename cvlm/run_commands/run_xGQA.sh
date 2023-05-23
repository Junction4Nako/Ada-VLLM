# raw ALBEF on GQA
python3 cvlm/run_xGQA_albef.py \
    --albef_config albef/configs/qa_base_xGQA.yaml \
    --model_name_or_path /remote-home/zjli/CVLM/ckpt/pretrained/ALBEF.pth \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/txt_models/bert-base-uncased \
    --do_train --do_lower_case --save_epoch 1 \
    --per_gpu_train_batch_size 32 --learning_rate 0.00003 \
    --per_gpu_eval_batch_size 128  --ans2label_map /remote-home/zjli/CVLM/datasets/GQA/label_map/trainval_all_ans2label.pkl \
    --num_train_epochs 30 --weight_decay 0.05 \
    --max_seq_length 40  --evaluate_during_training  --logging_steps 100 \
    --output_dir output_xGQA/raw_albef_en/  --cuda_devices 0,1,2,3,4,5,6

# xlm-r based stage 1
python3 cvlm/run_xGQA_albef.py \
    --albef_config albef/configs/qa_base_xGQA_xlm-r_freeze_single.yaml \
    --model_name_or_path pretrain/stage1_xlm-r_all_langs/checkpoint-0120000 \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r \
    --do_train --do_lower_case --save_epoch 1 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00003 \
    --per_gpu_eval_batch_size 32  --ans2label_map /remote-home/zjli/CVLM/datasets/GQA/label_map/trainval_all_ans2label.pkl \
    --num_train_epochs 30 --weight_decay 0.05  --test_lang all \
    --max_seq_length 40  --evaluate_during_training  --logging_steps 100 \
    --output_dir output_xGQA/stage1_xlmr_freeze_single/  --cuda_devices 0,1,2,3,4,5,6

# xlm-r based stage 2
python3 cvlm/run_xGQA_albef.py \
    --albef_config albef/configs/qa_base_xGQA_xlm-r_freeze_single.yaml \
    --model_name_or_path pretrain/stage2_xlmr_trans_all_lang_vis_freeze/checkpoint-0135000 \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r \
    --do_train --do_lower_case --save_epoch 1 \
    --per_gpu_train_batch_size 64 --learning_rate 0.00003 \
    --per_gpu_eval_batch_size 128  --ans2label_map /remote-home/zjli/CVLM/datasets/GQA/label_map/trainval_all_ans2label.pkl \
    --num_train_epochs 10 --weight_decay 0.05  --test_lang all \
    --max_seq_length 40  --evaluate_during_training  --logging_steps 20 \
    --output_dir output_xGQA/stage2_xlmr_freeze_single/  --cuda_devices 0,1,2,3,4,5

# unified stage 2 (MLM + TLM)
python3 cvlm/run_xGQA_albef.py \
    --albef_config albef/configs/qa_base_xGQA_xlm-r_freeze_single.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_128txt/checkpoint-0240000 \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r \
    --do_train --do_lower_case --save_epoch 1  --image_dir_format local \
    --per_gpu_train_batch_size 64 --learning_rate 0.00003 \
    --per_gpu_eval_batch_size 128  --ans2label_map /remote-home/zjli/CVLM/datasets/GQA/label_map/trainval_all_ans2label.pkl \
    --num_train_epochs 10 --weight_decay 0.05  --test_lang all \
    --max_seq_length 40  --evaluate_during_training  --logging_steps 20 \
    --output_dir output_xGQA/uni_stage2_xlmr_vlm_tlm_128mlm_freeze_single/  --cuda_devices 0,1,2,3,4,5


# for xlm-r init
python3 cvlm/run_xGQA_albef.py \
    --albef_config albef/configs/xGQA/qa_base_xGQA_xlm-r_init_freeze_single.yaml \
    --model_name_or_path pretrain/unified_stage2_xlm-r_all_langs_mono_init_8gpus/checkpoint-0240000 \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r \
    --do_train --do_lower_case --save_epoch 1  --image_dir_format local \
    --per_gpu_train_batch_size 64 --learning_rate 0.00003 \
    --per_gpu_eval_batch_size 128  --ans2label_map /remote-home/zjli/CVLM/datasets/GQA/label_map/trainval_all_ans2label.pkl \
    --num_train_epochs 10 --weight_decay 0.05  --test_lang all \
    --max_seq_length 40  --evaluate_during_training  --logging_steps 20 \
    --output_dir output_xGQA/uni_stage2_xlmr_vlm_tlm_mlm_init_freeze_single/ 

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

## do test
python3 cvlm/run_xGQA_albef.py \
    --albef_config albef/configs/qa_base_xGQA_xlm-r_freeze_single.yaml \
    --eval_model_dir output_xGQA/uni_stage2_xlmr_vlm_tlm_128mlm_freeze_single/checkpoint-0-1842/ \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r \
    --do_test --do_eval --do_lower_case --save_epoch 1  --image_dir_format local \
    --per_gpu_eval_batch_size 64  --ans2label_map /remote-home/zjli/CVLM/datasets/GQA/label_map/trainval_all_ans2label.pkl \
    --test_lang all  --eval_split test --max_seq_length 40 \
    --output_dir output_xGQA/evaluation/  --cuda_devices 0,1,2,3,4,5