#!/usr/bin/env bash

log=logger-mdrnn.log

echo "exp_dir" > $log
python trainmdrnn.py --logdir exp_dir --epochs 20

echo "factor_dir" > $log
python trainmdrnn.py --logdir factor_dir --epochs 20
