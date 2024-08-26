#!/usr/bin/env bash
cd ../..

# Train
CUDA_VISIBLE_DEVICES=3 python -m TBCQ.main_train --multirun  env=thickener save_dir=tbcq model=tbcq behavior_policy=pid train.ts=True train.tl=20 env.noise_type=2 env.ts=True
