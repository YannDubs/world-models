#!/usr/bin/env bash

log=logger.log

echo "exp_dir_nornn" > $log
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir_nornn --target-return 950 --display --max-epoch 51

echo "factor_dir_nornn" > $log
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir factor_dir_nornn --target-return 950 --display --max-epoch 51

echo "exp_dir" > $log
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir --target-return 950 --display --max-epoch 51

cp -r exp_dir exp_dir_gate
rm -rf exp_dir_gate/ctrl
rm -rf exp_dir_gate/tmp
echo "exp_dir_gate" > $log
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir_gate --is-gate --target-return 950 --display --max-epoch 51

echo "factor_dir" > $log
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir factor_dir --target-return 950 --display --max-epoch 51

cp -r exp_dir exp_dir_long
echo "long" > $log
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir_long --is-gate --target-return 950 --display --max-epoch 300
