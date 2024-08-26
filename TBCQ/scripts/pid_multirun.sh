#!/usr/bin/env bash
cd ../..

# Train
CUDA_VISIBLE_DEVICES=3 python -m TBCQ.main_train --multirun  env=thickener save_dir=pid model=pid env.noise_type=0 train.max_timesteps=1e5 random_seed=0
