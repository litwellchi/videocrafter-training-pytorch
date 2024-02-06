name="base_1024_t2v_v1"

# ckpt='/aifs4su/mmcode/videogen/share_ckpts/VideoCrafter/VideoCrafter2/model.ckpt'
ckpt='/aifs4su/mmcode/videogen/MACVideoGen/overfit_macvid_1/checkpoints/epoch=0008-step=000243.ckpt'
config='configs/inference_t2v_1024_v1.0.yaml'

# prompt_file="prompts/test_prompts.txt"
prompt_file="/aifs4su/mmcode/videogen/MACVideoGen/prompts/test_prompts.txt"
res_dir="results"

python3 scripts/evaluation/inference.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 4 --height 576 --width 1024 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 28