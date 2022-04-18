CUDA_VISIBLE_DEVICES=7 python main_cw.py \
  -run_type predict \
  -task e2e \
  -ckpt ./test/test_e2e_cpt/ckpt-epoch18 \
  -output preds \
  -batch_size 64 \
  -epochs 60






