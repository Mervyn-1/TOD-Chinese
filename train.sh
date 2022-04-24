CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 --master_port 21116  train.py \
  -run_type train \
  -task e2e \
  -max_to_keep_ckpt 70 \
  -learning_rate 5e-5 \
  -model_dir ./test/test_e2e_cpt_base \
  -batch_size 10 \
  -epochs 70 \
  -backbone fnlp/cpt-base
