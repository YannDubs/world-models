#!/usr/bin/env bash

log=logger_all.log

#echo "vae" >> $log
#python trainvae.py --logdir all_dir --is-factor --batch-size 256 --epochs 200 --max-workers 16

#echo "mdrnn" >> $log
#python trainmdrnn.py --logdir all_dir --no-train

echo "control" >> $log
xvfb-run -a -s "-screen 0 1400x900x24" python traincontroller.py --logdir all_dir --n-samples 4 --pop-size 4 --max-workers 16 --target-return 950 --is-gate --max-epoch 150 --display

cp -r all_dir all_dir_mdrnn
echo "all_dir_mdrnn" >> $log

echo "mdrnn" >> $log
python trainmdrnn.py --logdir all_dir_mdrnn

echo "control" >> $log
xvfb-run -a -s "-screen 0 1400x900x24" python traincontroller.py --logdir all_dir_mdrnn --n-samples 4 --pop-size 4 --max-workers 16 --target-return 950 --is-gate --max-epoch 50 --display

echo "back to factor no rnn" >> $log

echo "vae" >> $log
python trainvae.py --logdir all_dir --is-factor --batch-size 256 --epochs 100 --max-workers 16

xvfb-run -a -s "-screen 0 1400x900x24" python traincontroller.py --logdir all_dir --n-samples 4 --pop-size 4 --max-workers 16 --target-return 950 --is-gate --max-epoch 10000 --display
