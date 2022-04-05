CUDA_VISIBLE_DEVICES=5 python main_cw.py \
  -run_type predict \
  -task e2e \
  -ckpt ./test_e2e/ckpt-epoch34 \
  -output preds \
  -batch_size 64 \
  -epochs 60






