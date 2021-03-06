""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
parser.add_argument('--is-gate', action='store_true', help='Whether to use a highway for last actions to smoothen it out.')
args = parser.parse_args()

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 1000, is_gate=args.is_gate)

with torch.no_grad():
    print(generator.rollout(None, render=True))
