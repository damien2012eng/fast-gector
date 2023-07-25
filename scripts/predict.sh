#!/bin/bash
base_dir="/home/ec2-user/gector"
mkdir result
python predict.py \
    --batch_size 256 \
    --iteration_count 5 \
    --min_len 3 \
    --max_len 128 \
    --min_error_probability 0.0 \
    --additional_confidence 0.0 \
    --sub_token_mode "average" \
    --max_pieces_per_token 5 \
    --model_dir ${base_dir} \
    --ckpt_id "epoch-5" \
    --detect_vocab_path "./data/vocabulary/d_tags.txt" \
    --correct_vocab_path "./data/vocabulary/labels.txt" \
    --pretrained_transformer_path "${base_dir}/roberta_1_gectorv2.th" \
    --input_path "${base_dir}/determiners-alternatives/sents/sentences_sample_100.txt" \
    --out_path "result/sentences_sample_100.pred" \
    --special_tokens_fix 1 \
    --detokenize 1 \
    --segmented 1
