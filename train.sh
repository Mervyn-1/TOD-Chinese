CUDA_VISIBLE_DEVICES=5 python main_cw.py \
  -run_type train \
  -task e2e \
  -max_to_keep_ckpt 60 \
  -learning_rate 9e-5 \
  -model_dir ./test/test_e2e_bart\
  -batch_size 10 \
  -epochs 60 \
  -backbone fnlp/bart-base-chinese

