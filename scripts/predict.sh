#!/bin/bash
pretrained_transformer_path="roberta-base"
ckpt_path="/home/ec2-user/gector/roberta_1_gectorv2.th"
input_path="/home/ec2-user/fast-gector/test.src"
out_path="/home/ec2-user/fast-gector/test.pred"
python predict.py \
    --batch_size 128 \
    --iteration_count 1 \
    --min_seq_len 3 \
    --max_num_tokens 128 \
    --min_error_probability 0.0 \
    --additional_confidence 0.0 \
    --sub_token_mode "average" \
    --max_pieces_per_token 5 \
    --ckpt_path $ckpt_path \
    --detect_vocab_path "./data/vocabulary/d_tags.txt" \
    --correct_vocab_path "./data/vocabulary/labels.txt" \
    --pretrained_transformer_path $pretrained_transformer_path \
    --input_path $input_path \
    --out_path $out_path \
    --special_tokens_fix 1 \
    --detokenize 0 \
    --segmented 1
