name="base_1024_test"

conda activate videocrafter
ckpt='ckpts/Text2Video576x1024/model.ckpt'
config='configs/inference_t2v_1024_v1.0.yaml'

prompt_file="test_demo/demo_prompts/crocodile.txt"
res_dir="results"

python3 scripts/evaluation/inference.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 576 --width 1024 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 28