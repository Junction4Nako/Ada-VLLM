python3 cvlm/run_xGQA_albef.py \
    --albef_config albef/configs/qa_base_vgqa_ja_xlm-r.yaml \
    --model_name_or_path pretrain/stage1_xlm-r_all_langs/checkpoint-0120000 \
    --tokenizer_name /remote-home/zjli/CVLM/ckpt/pretrained/xlm-r \
    --do_train --do_lower_case --save_epoch 1 \
    --per_gpu_train_batch_size 16 --learning_rate 0.00003 \
    --per_gpu_eval_batch_size 32  --ans2label_map /remote-home/zjli/CVLM/datasets/vqa_vg_ja/ans2label_ja.pkl \
    --num_train_epochs 30 --weight_decay 0.05  --test_lang all \
    --max_seq_length 40  --evaluate_during_training  --logging_steps 100 \
    --output_dir output_vgqa/stage1_xlmr/  --cuda_devices 0,1,2,3,4,5,6