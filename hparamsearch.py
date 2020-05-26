import sys
import os
import time
import argparse
import itertools
import math
import pprint
import numpy
import random

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import torchtext.data as data 
import torchtext.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import load_data

from rnnlm import RNNLM
from sentence_vae import SentenceVAE

from train import *

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--embed_dim', type=int, default=300, help="The amount of dimensions of the word embeddings.")
    parser.add_argument('--hidden_dim', type=int, default=512, help="The amount of hidden dimensions.")
   
    parser.add_argument('--freebits_lambda', type=float, default=0, help="")
    parser.add_argument('--wdropout_prob', type=float, default=1, help="")
    parser.add_argument('--mu_forcing_beta', type=float, default=0, help="")

    # Training params
    parser.add_argument('--use_bptt', type=bool, default=False, help='')
    parser.add_argument('--seq_len', type=int, default=int(25), help='The length of the sequences to train on.')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples to process in a batch')
    parser.add_argument('--device', type=str, default="cuda", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.95, help='Learning rate decay fraction')
    parser.add_argument('--train_steps', type=int, default=int(14000), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='Gradient clipping maximum norm.')

    # Misc params
    parser.add_argument('--print_every', type=int, default=100, help='How often to print training progress')
    parser.add_argument('--sw_log_dir', type=str, default='hparamruns', help='The directory in which to create the default logdir.')
    parser.add_argument('--save_file', type=str, default='model.pt', help='Filename under which to store the model.')

    config = parser.parse_args()
    config.model = 's-vae'

    torch.manual_seed(42)
    numpy.random.seed(42)
    random.seed(42)
    
    def train_model(config):
        # summarywriter 
        logdir = logloc(dir_name=config.sw_log_dir, comment="")
        sw = SummaryWriter(log_dir=logdir)

        # Train and save the model
        best_model, model, metrics = train(config, sw)
        model = model.cpu()

        config_dict = dict(vars(config))
        for key in ['use_bptt','seq_len','device','train_steps','print_every','sw_log_dir','save_file']:
            del config_dict[key]

        sw.add_hparams(config_dict, metrics)
        sw.close()

        save_model("last", model, config)

    #train_model(config)

    config.wdropout_prob = 0.55

    #for lambd in [10]:
    #    config.freebits_lambda = lambd / 512
    #    train_model(config)
    #config.freebits_lambda = 0

    for beta in [20]:
        config.mu_forcing_beta = beta
        train_model(config)

    #for wdrop in [0.4, 0.45, 0.5, 0.55, 0.6]:
    #    config.wdropout_prob = wdrop
    #    train_model(config)
    #config.wdropout_prob = 1

    
