CUDA_VISIBLE_DEVICES=0 python main_cw.py \
  -run_type train \
  -task e2e \
  -max_to_keep_ckpt 60 \
  -learning_rate 1.2e-4 \
  -model_dir ./test_e2e_bart\
  -batch_size 4 \
  -epochs 60 \
  -backbone fnlp/bart-base-chinese


