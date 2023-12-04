import os
import random
import argparse
import configparser

import numpy as np
from torch.backends import cudnn
import torch

from solver import Solver
import sys


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    else:
        solver.train()
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    args = parser.parse_args()
    fileconfig = configparser.ConfigParser()
    fileconfig.read(args.config)

    parser.add_argument('--lr', type=float, default=fileconfig['train']['lr'])
    parser.add_argument('--gpu', type=str, default=fileconfig['train']['gpu'])
    parser.add_argument('--num_epochs', type=int, default=fileconfig['train']['epoch'])
    parser.add_argument('--anormly_ratio', type=float, default=fileconfig['train']['ar'])
    parser.add_argument('--batch_size', type=int, default=fileconfig['train']['bs'])
    parser.add_argument('--seed', type = int, default = fileconfig['train']['seed'])

    parser.add_argument('--win_size', type=int, default=fileconfig['data']['ws'])
    parser.add_argument('--input_c', type=int, default=fileconfig['data']['ic'])
    parser.add_argument('--output_c', type=int, default=fileconfig['data']['oc'])
    parser.add_argument('--dataset', type=str, default=fileconfig['data']['ds'])
    parser.add_argument('--data_path', type=str, default=fileconfig['data']['dp'])
    
    parser.add_argument('--d_model', type=int, default=fileconfig['param']['d'])
    parser.add_argument('--e_layers', type=int, default=fileconfig['param']['l'])
    parser.add_argument('--fr', type=float, default=fileconfig['param']['fr'])
    parser.add_argument('--tr', type=float, default=fileconfig['param']['tr'])
    parser.add_argument('--seq_size', type=int, default=fileconfig['param']['ss'])

    parser.add_argument('--mode', type=str, default=fileconfig['model']['mode'])
    parser.add_argument('--model_save_path', type=str, default=fileconfig['model']['msp'])
    

    config = parser.parse_args()

    args = vars(config)

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True

    sys.stdout = Logger("result/"+ config.dataset +".log", sys.stdout)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
