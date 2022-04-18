CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 train.py \
  -run_type train \
  -task e2e \
  -max_to_keep_ckpt 60 \
  -learning_rate 5e-5 \
  -model_dir ./test/test_e2e_cpt_ddp \
  -batch_size 10 \
  -epochs 60 \
  -backbone fnlp/cpt-base