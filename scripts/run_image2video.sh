name="i2v_512_test"

ckpt='ckpts/Image2Video320x512/model.ckpt'
config='configs/inference_i2v_512_v1.0.yaml'

prompt_file="test_demo/demo_prompts/dragon.txt"
condimage_dir="test_demo/demo_image"
res_dir="results"

python3 scripts/evaluation/inference.py \
--seed 123 \
--mode 'i2v' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--cond_input $condimage_dir \
--fps 8

