CUDA_VISIBLE_DEVICES=0 python train.py \
  -run_type predict \
  -task e2e \
  -model_dir ./test/test_e2e_cpt_ddp/2022-04-17-23_03_29_257667/model-epoch_15-batch_31770-combined_score#gen_102.39956345746552 \
  -output pred_output





