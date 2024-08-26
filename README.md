# TBCQ
This repo is the official implementation for Controlling Partially Observed Industrial System based on Offline Reinforcement Learning - A Case Study of Paste Thickener.

**[Zhaolin Yuan](), [Zixuan Zhang](), [Xiaorui Li](), [Yunduan Cui](), [Ming Li](), [Xiaojuan Ban]()**

**TII 2024**

### Abstract 
In the field of mineral processing, controlling the paste thickener is a highly challenging and critical task because of the high complexity, incomplete observation space, and excessive environmental noises.
The paper proposes an offline-data-driven controlling strategy to optimize the operational indices in the thickening system based on offline reinforcement learning (RL).
Compared to common reinforcement learning methods that rely on online interactive training, our approach ensures the safety of the production process by training the controller solely using offline datasets, thereby avoiding dangerous online exploration.
In terms of offline dataset collection, this study utilizes the prior knowledge of the thickening mechanism to design a PID controller as the behavior policy to collect operational trajectories as the offline dataset.
Additionally, to tackle a critical issue in controlling the thickening system: constrained observation space, this paper analyzes the dynamical properties of the thickening system and introduces a novel offline reinforcement learning algorithm, Temporal Batch-constrained Q-learning (TBCQ).
The algorithm and associated model framework are specifically developed for controlling Partially Observed Markov Decision Processes (POMDP).
The TBCQ and trained policy are evaluated in both a simulated thickening environment and a real industrial paste thickener in a copper mine.
The real-world experiments demonstrate that the proposed controller outperforms the baselines and effectively reduces the tracking error of underflow concentration by over 12\%.
The successful application of our pipeline in paste thickener also offers an innovative perspective on addressing optimization problems in complex industrial systems: performing offline reinforcement learning on a dataset sampled from a suboptimal policy.


### Citation
If you use this dataset for your research, please cite our paper:


### QuickStart
Our code is built off of the BCQ[https://github.com/sfujim/BCQ] repository and uses many similar components. 

To run TBCQ, please use a command like this:
```
python -m TBCQ.main_train --multirun save_dir=tbcq env=thickener model=tbcq behavior_policy=pid 
env.noise_type=2 env.ts=False train.ts=True train.tl=20 train.buffer_size=5e3 random_seed=0
```
Before conducting the multi random seed traversal test, it is necessary to manually place the model files 
stored in the ```ckpt``` folder into the ```model_file``` folder. And rename the model file 
as ```noise-2_tbcq-pid_buffer-5000_random-0_traj-1.pkl```. Then use a command list this:
```
python -m TBCQ.main_test --multirun save_dir=tbcq_test env=thickener model=tbcq behavior_policy=pid env.noise_type=2 
env.ts=False train.ts=True train.tl=20 train.buffer_size=5e3 random_seed=0
```

