## 这玩意就是VideoCrafter但是咱们训个更牛逼的
焕发HKUST新荣光


## 🤗 Acknowledgements
Our codebase builds on [Stable Diffusion](https://github.com/Stability-AI/stablediffusion). 
Our codebase builds on [VideoCrafter](https://github.com/litwellchi/VideoCrafter.git). 
Thanks the authors for sharing their awesome codebases! 

## Install
Already adjust requirements.txt to H800.
```shell
pip install -r requirements.txt
```
## Data 
```
/scratch/suptest/video_data/macvid/
```
数据还在这个位置下；有一些包的metadata还没ready


## Training
```shell
source activate anaconda3/bin/activate
conda activate videocrafter
sh scripts/train_diffusion.sh
```
数据的config形式变换为在`/home/xchiaa/MACVideoGen/configs/training_data`设置要跑的文件。
细节懒得写了，下次一定.
目前由于集群问题，所以wandb需要手动同步一下`wandb sync test_macvid_t2v_1024_20240207/wandb/offline-run-20240207_064351-test_macvid_t2v_1024_20240207`