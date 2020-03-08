python train.py \
    --batch_size 10 \
    --caption_model transformer \
    --input_att_dir '/data/VG/parabu_att' \
    --input_fc_dir '/data/VG/parabu_fc' \
    --input_json 'data/paratalk.json' \
    --input_label_h5 'data/paratalk_label.h5' \
    --language_eval 1 \
    --learning_rate 5e-4 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --max_epochs 100 \
    --rnn_type 'lstm' \
    --val_images_use 2486 \
    --save_checkpoint_every 2000 \
    --checkpoint_path 'log_att_xe/' \
    --id trans_att_mask \
    --print_freq 200 \
    --seq_per_img 7 \
    --max_length 30 \
    #--start_from pretrained_coco